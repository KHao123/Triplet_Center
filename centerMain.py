import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets import UnbalancedMNIST, BalancedBatchSampler
from networks import EmbeddingNet, ClassificationNet,ResNetEmbeddingNet
from skinDatasetFolder import skinDatasetFolder
from covidDataSetFolder import CovidDataset
from losses import OnlineTripletLoss,OnlineContrastiveLoss,OnlineCenterLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from utils import BatchHardTripletSelector,AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch
from trainer import fit
from CenterLoss import CenterLoss
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from trainer import getMetrics,plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score ,accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import csv

Name_dict = {
    'MNIST' : ['0','1','2','3','4','5','6','7','8','9'],
    'skin' : ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
    'Retina' : ['0','1','2','3','4'],
    'Xray14': ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass',
    'No_Finding','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'],
    'xray3' : ['Normal', 'Lung Opacity', '‘No Lung Opacity/Not Normal'],
	'covid19' : ['Normal','covid19','Others']
}

def str2bool(v):
    """Convert string to Boolean

    Args:
        v: True or False but in string

    Returns:
        True or False in Boolean

    Raises:
        TyreError
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Triplet For MNIST')
parser.add_argument('--dataset_name',default='covid19',
					help='Choose dataset [...]')
parser.add_argument('--rescale',default=False,type=str2bool,
					help='rescale dataset')
parser.add_argument('--iterNo',default=1,type=int,
					help='Use for choosing fold validation')
parser.add_argument('--cuda_device',default=0,type=int,
					help='Choose cuda_device:(0,1,2,3,4,5,6,7)')
parser.add_argument('--EmbeddingMode',default=False,type = str2bool ,
					help='True for tripletsLoss(embedding) / False for EntropyLoss(classfication)')
parser.add_argument('--dim',default=128,type=int,
					help='The dimension of embedding(type int)')
parser.add_argument('--n_classes',default=3,type=int,
					help='The number of classes (type int)')
parser.add_argument('--margin',default=0.5,type=float,
					help='Margin used in triplet loss (type float)')
parser.add_argument('--logdir',default='result',
					help='Path to log tensorboard, pick a UNIQUE name to log')
parser.add_argument('--start_epoch',default=0,type=int
					,help='Start epoch (int)')
parser.add_argument('--n_epoch',default=200,type=int,
					help='End_epoch (int)')
parser.add_argument('--batch_size',default=16,type=int,
					help='Batch size (int)')
parser.add_argument('--n_sample_classes',default=3,type=int,
				help='For a batch sampler to work comine #samples_per_class')
parser.add_argument('--n_samples_per_class',default=10,type=int,
				help='For a batch sampler to work comine #n_sample_classes')
parser.add_argument('--TripletSelector',default='SemihardNegativeTripletSelector',
					help='Triplet selector chosen in ({},{},{},{},{})'
					.format('AllTripletSelector',
						'HardestNegativeTripletSelector',
						'RandomNegativeTripletSelector',
						'SemihardNegativeTripletSelector',
						'BatchHardTripletSelector'))
args = parser.parse_args()

def extract_embeddings(dataloader, model, dimension):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dimension))#num_of_dim
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def calc_centers(dataloader,model,n_classes):
	with torch.no_grad():
		model.eval()
		centers = torch.tensor([],requires_grad=True).cuda()
		embeddings = torch.Tensor([]).cuda()
		targets = torch.LongTensor([]).cuda()
		for (data,target) in dataloader:
			data , target = data.cuda() , target.cuda()
			batch_embedding = model(data)
			embeddings = torch.cat([embeddings,batch_embedding])
			targets = torch.cat([targets,target])

	for lbl in range(n_classes):
		mask = targets.eq(lbl)
		embeddings_ = embeddings[mask]
		center = embeddings_.mean(dim=0)
		centers = torch.cat([centers,center.unsqueeze(dim=0)])
	assert centers.shape == (n_classes,embeddings.size()[1])
	#print(centers,centers.size(),centers.requires_grad)
	return centers

def CenterPredict(embeddings,centers):
    C = centers.size()[0]
    n = embeddings.size()[0]
    labels = torch.LongTensor([])
    for i in range(n):
        dis = (embeddings[i].repeat((C,1))-centers).pow(2).sum(1)
        labels = torch.cat([labels,torch.LongTensor([torch.min(dis,0)[1]])])
    return labels

def GetMetric(y_true,y_pred):
	accuracy = accuracy_score(y_true,y_pred)
	precision = precision_score(y_true,y_pred,average=None)
	recall = recall_score(y_true,y_pred,average=None)
	MCA , MCR = precision.mean() , recall.mean()
	print(precision,recall)
	return accuracy , MCA , MCR

if __name__ == '__main__':
	torch.cuda.set_device(args.cuda_device)
	dataset_name = args.dataset_name
	train_dataset = CovidDataset(iterNo=args.iterNo,train=True)
	test_dataset = CovidDataset(iterNo=args.iterNo,train=False)

	
	cuda = torch.cuda.is_available()

	kwargs = {'num_workers': 40, 'pin_memory': True} if cuda else {}
	batch_size = args.batch_size
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


	start_epoch = args.start_epoch
	n_epochs = args.n_epoch
	n_classes = args.n_classes
	margin = args.margin
	


	embedding_net = ResNetEmbeddingNet(dataset_name,args.dim)
	#classification_net = ClassificationNet(embedding_net, dimension = num_of_dim ,n_classes = n_classes)
	device = torch.device("cuda")
	model = embedding_net
	pth = './{}_d-checkpoint/iterNo{}.pth'.format(args.dim,args.iterNo)
	location = 'cuda:{}'.format(args.cuda_device)
	model.load_state_dict(torch.load(pth, map_location=location),strict=False)
	model.to(device)

	print('Check point loaded successfully! '+ pth)
	# loss_fn = CenterLoss(margin,n_classes)
	loss_fn = OnlineCenterLoss(margin)
	# loss_fn = OnlineCenterLoss(margin)
	lr = 1e-4

	optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.99),
                eps=1e-8,
                amsgrad=True)
	scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)

	output_writer_path = os.path.join('./run/result', '{}Res/iterNo{}/Centerloss_margin{}_{}d'.format(
			dataset_name, args.iterNo, margin,args.dim)
		)
	checkpoint_path = output_writer_path
	writer = SummaryWriter(output_writer_path)
	csvfileName = os.path.join(output_writer_path,'result.csv')

	best_mf1 = 0
    ###################################
	with open(csvfileName,'w',newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		Metrics = ['_p','_r','_f1']
		firstRow = []
		for name in Name_dict[args.dataset_name]:
			for m in Metrics:
				firstRow.append(name+m)
		firstRow.extend(['mean_p','mean_r','mean_f1'])
		csvwriter.writerow(firstRow)

		for epoch in range(start_epoch,n_epochs):
			scheduler.step()

			centers = calc_centers(train_loader,model,n_classes)
			centers.requires_grad_()
			
			model.train()
			Loss = 0
			correct_train = []
			pred_train = []
			for batch_idx,(data,target) in enumerate(train_loader):
				optimizer.zero_grad()
				data , target = data.cuda() , target.cuda()
				batch_embedding = model(data)
				loss = loss_fn(batch_embedding,target,centers)
				loss.backward()
				optimizer.step()
				Loss += loss.item()

				correct_train.extend(target.data.cpu().numpy())
				pred_train.extend(CenterPredict(batch_embedding,centers).data.cpu().numpy())
				
				if batch_idx%10 == 0:
					print('Train => epoch:{}  |  batch:{} | loss:{}'.format(epoch,batch_idx,loss.item()))
			train_acc, train_precision, train_recall, train_f1, train_mca, train_mcr, train_mf1 = \
															getMetrics(dataset_name,correct_train, pred_train)
			writer.add_scalar('train/loss', Loss, epoch)
			writer.add_scalar('train/acc',train_acc,epoch)
			writer.add_scalar('train/mca',train_mca,epoch)
			writer.add_scalar('train/mcr',train_mcr,epoch)
			writer.add_scalar('train/mean_f1',train_mf1,epoch)
			for lbl,precision_ in enumerate(train_precision):
				writer.add_scalar('train/class_precision_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),precision_,epoch)
			for lbl,recall_ in enumerate(train_recall):
				writer.add_scalar('train/class_recall_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),recall_,epoch)
			for lbl,f1_ in enumerate(train_f1):
				writer.add_scalar('train/class_f1_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),f1_,epoch)

			print('Train => [epoch : {} | Loss : {} | ACC :{} | MCA : {} | MCR : {} | Mf1 : {} ]'.format(epoch,Loss,train_acc,train_mca,train_mcr,train_mf1))

			with torch.no_grad():
				model.eval()
				# centers = calc_centers(test_loader,model,n_classes)
				#centers.requires_grad_()
				Loss = 0
				correct_test = []
				pred_test = []
				for batch_idx,(data,target) in enumerate(test_loader):
					data , target = data.cuda() , target.cuda()
					batch_embedding = model(data)
					loss = loss_fn(batch_embedding,target,centers)
					Loss += loss.item()
					correct_test.extend(target.data.cpu().numpy())
					pred_test.extend(CenterPredict(batch_embedding,centers).data.cpu().numpy())
					
					if batch_idx%10 == 0:
						print('Test => epoch:{}  |  batch:{} | loss:{}'.format(epoch,batch_idx,loss.item()))
				
				test_acc, test_precision, test_recall, test_f1, test_mca, test_mcr, test_mf1 = \
													 getMetrics(dataset_name,correct_test, pred_test)
				epochRow = []
				for i in range(len(Name_dict[args.dataset_name])):
					epochRow.extend([test_precision[i],test_recall[i],test_f1[i]])
				epochRow.extend([test_mca,test_mcr,test_mf1])
				csvwriter.writerow(epochRow)

				writer.add_scalar('test/loss', Loss, epoch)
				writer.add_scalar('test/acc',test_acc,epoch)
				writer.add_scalar('test/mca',test_mca,epoch)
				writer.add_scalar('test/mcr',test_mcr,epoch)
				writer.add_scalar('test/mean_f1',test_mf1,epoch)
				for lbl,precision_ in enumerate(test_precision):
					writer.add_scalar('test/class_precision_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),precision_,epoch)
				for lbl,recall_ in enumerate(test_recall):
					writer.add_scalar('test/class_recall_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),recall_,epoch)
				for lbl,f1_ in enumerate(test_f1):
					writer.add_scalar('test/class_f1_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),f1_,epoch)



				if test_mf1>best_mf1:
					best_mf1 = test_mf1
					torch.save(model.state_dict(),checkpoint_path+'/Mf1-{:.4f}'.format(best_mf1)+'.pth')
					print('*************** Best_f1 Log ***************\nMf1-{:.4f}\tMCA-{:.4f}\tMCR-{:.4f}'.format(test_mf1 , test_mca , test_mcr))
					fig, title = plot_confusion_matrix( dataset_name, correct_test, pred_test,False)
					plt.close()
					writer.add_figure(title, fig, epoch)

				print('Test => [epoch : {} | Loss : {} | ACC :{} | MCA : {} | MCR : {} | Mf1 : {} ]'.format(epoch,Loss,test_acc,test_mca,test_mcr,test_mf1))

				if epoch%5 == 0:
					test_data_embeddings , correct = extract_embeddings(test_loader,model,args.dim)
					writer.add_embedding(test_data_embeddings,
	                    metadata = correct,
	                    global_step = epoch
	                )

				if epoch+1 == n_epochs:
					print('*************** End_epoch Log ***************\nMf1-{:.4f}\tMCA-{:.4f}\tMCR-{:.4f}'.format(test_mf1 , test_mca , test_mcr))