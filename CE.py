import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from datasets import UnbalancedMNIST, BalancedBatchSampler
from networks import EmbeddingNet, ClassificationNet,ResNetEmbeddingNet
from metrics import AccumulatedAccuracyMetric,AverageNonzeroTripletsMetric
from skinDatasetFolder import skinDatasetFolder
from covidDataSetFolder import CovidDataset
from losses import OnlineTripletLoss,OnlineContrastiveLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from utils import BatchHardTripletSelector,AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch
from trainer import fit

import torch
from torch.optim import lr_scheduler
import torch.optim as optim


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
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


if __name__ == '__main__':
	print(args)

	torch.cuda.set_device(args.cuda_device)
	logdir = args.logdir

	dataset_name = args.dataset_name
	
	Attr_Dict = {
	'covid19':{'in_channel':1,
			'n_classes':3,
			'train_dataset' : CovidDataset(iterNo=args.iterNo,train=True),
			'test_dataset' : CovidDataset(iterNo=args.iterNo,train=False),
			'resDir':'./covid19Res/iterNo{}'.format(args.iterNo)
			}
	}

	num_of_dim = args.dim
	n_classes = Attr_Dict[dataset_name]['n_classes']	
	train_dataset = Attr_Dict[dataset_name]['train_dataset']
	test_dataset = Attr_Dict[dataset_name]['test_dataset']
	
	n_sample_classes = args.n_sample_classes
	n_samples_per_class = args.n_samples_per_class
	train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=n_sample_classes, n_samples=n_samples_per_class)
	test_batch_sampler = BalancedBatchSampler(test_dataset,  n_classes=n_sample_classes, n_samples=n_samples_per_class)

	cuda = torch.cuda.is_available()

	kwargs = {'num_workers': 40, 'pin_memory': True} if cuda else {}
	batch_size = args.batch_size
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

	online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
	online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)


	start_epoch = args.start_epoch
	n_epochs = args.n_epoch
	log_interval = 50
	margin = args.margin

	Selector = {
		'AllTripletSelector':AllTripletSelector(),
		'HardestNegativeTripletSelector':HardestNegativeTripletSelector(margin),
		'RandomNegativeTripletSelector':RandomNegativeTripletSelector(margin),
		'SemihardNegativeTripletSelector':SemihardNegativeTripletSelector(margin),
		'BatchHardTripletSelector':BatchHardTripletSelector(margin)
	}

	embedding_net = ResNetEmbeddingNet(dataset_name,num_of_dim)
	classification_net = ClassificationNet(embedding_net, dimension = num_of_dim ,n_classes = n_classes)
	
	if args.EmbeddingMode:
		loader1 = online_train_loader
		loader2 = online_test_loader
		model = embedding_net
		loss_fn = OnlineTripletLoss(margin, Selector[args.TripletSelector])
		lr = 1e-4
		# optimizer = optim.Adam(model.parameters(), lr=lr)
		optimizer = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.99),
                    eps=1e-8,
                    amsgrad=True)
		scheduler = lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
		metrics = [AverageNonzeroTripletsMetric()]
		logName = 'margin{}_{}d-embedding_{}'.format(margin,num_of_dim,args.TripletSelector)
		logName = os.path.join(Attr_Dict[dataset_name]['resDir'],logName)
		EmbeddingArgs = (num_of_dim,train_loader,test_loader)

	else:
		loader1 = train_loader
		loader2 = test_loader
		model = classification_net
		loss_fn = torch.nn.CrossEntropyLoss()
		lr = 1e-4
		# optimizer = optim.Adam(model.parameters(), lr=lr)
		optimizer = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    betas=(0.9, 0.99),
                    eps=1e-8,
                    amsgrad=True)
		scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
		metrics = [AccumulatedAccuracyMetric()]
		logName = '{}d-CE-no_class_weights'.format(num_of_dim)
		logName = os.path.join(Attr_Dict[dataset_name]['resDir'],logName)
		EmbeddingArgs = ()





	if cuda:
		model.cuda()

	logfile = os.path.join(logdir,logName)
	fit(dataset_name,
		logfile,
		loader1,
		loader2,
		model,
		loss_fn,
		optimizer,
		scheduler,
		n_epochs,
		cuda,
		log_interval,
		metrics,
		start_epoch,
		*EmbeddingArgs)


