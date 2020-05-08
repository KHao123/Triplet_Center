import torch
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
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

param_grid =[
    {
        'weights':['uniform'],
        'n_neighbors': [1,3,5],
        'p': [2]
    }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf,param_grid,cv=4,scoring='f1_macro')

cuda = torch.cuda.is_available()

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

def fit(dataset_name, logName, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, *EmbeddingArgs):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    if len(EmbeddingArgs) != 0:
        assert len(EmbeddingArgs)==3
        Embedding_Mode = True
        n_dim = EmbeddingArgs[0]
        all_train_loader = EmbeddingArgs[1]
        all_test_loader = EmbeddingArgs[2]
        train_Args = (n_dim,all_train_loader)
        test_Args = (n_dim,all_test_loader)
    else:
        Embedding_Mode = False
        train_Args = ()
        test_Args = ()

    output_writer_path = os.path.join('./run', logName)
    checkpoint_path = output_writer_path
    csvfileName = os.path.join(output_writer_path,'result.csv')
    writer = SummaryWriter(output_writer_path)

    Best_f1 = 0.0
    with open(csvfileName,'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        Metrics = ['_p','_r','_f1']
        firstRow = []
        for name in Name_dict[dataset_name]:
            for m in Metrics:
                firstRow.append(name+m)
        firstRow.extend(['mean_p','mean_r','mean_f1'])
        csvwriter.writerow(firstRow)
        for epoch in range(0, start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, n_epochs):
            scheduler.step()

            # Train stage
            train_loss, metrics = train_epoch(dataset_name,epoch,writer,train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics,*train_Args)
            writer.add_scalar('train/loss', train_loss, epoch)
            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            val_loss, metrics, test_f1 , test_mca , test_mcr = test_epoch(dataset_name,epoch,csvwriter,writer,val_loader, model, loss_fn, cuda, metrics,*test_Args)
            if test_f1 > Best_f1:
                Best_f1 = test_f1
                torch.save(model.state_dict(),checkpoint_path+'/Mf1-{:.4f}'.format(Best_f1)+'.pth')
                print('*************** Best_f1 Log ***************\nMf1-{:.4f}\tMCA-{:.4f}\tMCR-{:.4f}'.format(test_f1 , test_mca , test_mcr))

            if epoch+1 == n_epochs:
                print('*************** End_epoch Log ***************\nMf1-{:.4f}\tMCA-{:.4f}\tMCR-{:.4f}'.format(test_f1 , test_mca , test_mcr))

            val_loss /= len(val_loader)
            #######summary_writer#########
            #necessary for test_loss{classification or triplet loss}
            writer.add_scalar('test/loss', train_loss, epoch)
            ####
            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                     val_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)


def train_epoch(dataset_name,epoch,writer,train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,*EmbeddingArgs):
    if len(EmbeddingArgs) != 0:
        assert len(EmbeddingArgs)==2
        Mode = True
        n_dim = EmbeddingArgs[0]
        all_train_loader = EmbeddingArgs[1]
    else:
        Mode = False

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    correct = []
    predicted = []

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if Mode==False:
            _, pred = torch.max(outputs[0].data, 1)
            correct.extend(target[0].cpu().numpy())
            predicted.extend(pred.cpu().numpy())
            
        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []
    if Mode:
        ###To do correct = [] , predicted = []
        train_data_embeddings , correct = extract_embeddings(all_train_loader,model,n_dim)
        grid_search.fit(train_data_embeddings,correct)
        predicted = grid_search.best_estimator_.predict(train_data_embeddings)
        ###
    acc, precision, recall, f1, mca, mcr, m_f1 = getMetrics(dataset_name,correct, predicted)
    print('--Train ==> ACC:{}\tMCA:{}\tMCR:{}\tMeanF1:{}'.format(acc,mca,mcr,m_f1))
    ####summary_writer TO DO
    writer.add_scalar('train/mca', mca, epoch)
    writer.add_scalar('train/acc', acc, epoch)
    writer.add_scalar('train/mcr', mcr, epoch)
    writer.add_scalar('train/mean_f1',m_f1,epoch)
    for lbl,precision_ in enumerate(precision):
        writer.add_scalar('train/class_precision_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),precision_,epoch)
    for lbl,recall_ in enumerate(recall):
        writer.add_scalar('train/class_recall_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),recall_,epoch)
    for lbl,f1_ in enumerate(f1):
        writer.add_scalar('train/class_f1_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),f1_,epoch)
    #
    ####
    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(dataset_name,epoch,csvwriter,writer,val_loader,model, loss_fn, cuda, metrics,*EmbeddingArgs):
    if len(EmbeddingArgs) != 0:
        assert len(EmbeddingArgs)==2
        Mode = True
        n_dim = EmbeddingArgs[0]
        all_test_loader = EmbeddingArgs[1]
    else:
        Mode = False

    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        correct = []
        predicted = []

        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            if Mode==False:
                _, pred = torch.max(outputs[0].data, 1)
                correct.extend(target[0].cpu().numpy())
                predicted.extend(pred.cpu().numpy())

            for metric in metrics:
                metric(outputs, target, loss_outputs)

        if Mode:
            test_data_embeddings , correct = extract_embeddings(all_test_loader,model,n_dim)
            predicted = grid_search.best_estimator_.predict(test_data_embeddings)
            if epoch%5==0:
                writer.add_embedding(test_data_embeddings,
                    metadata = correct,
                    global_step = epoch
                )

        acc, precision, recall, f1, mca, mcr, m_f1 = getMetrics(dataset_name, correct, predicted)
        epochRow = []
        for i in range(len(Name_dict[dataset_name])):
            epochRow.extend([precision[i],recall[i],f1[i]])
        epochRow.extend([mca,mcr,m_f1])
        csvwriter.writerow(epochRow)

        print('--Test ==> ACC:{}\tMCA:{}\tMCR:{}\tMeanF1:{}'.format(acc,mca,mcr,m_f1))
  
        if epoch%5==0:
            fig, title = plot_confusion_matrix(dataset_name,correct, predicted,False)
            plt.close()
            writer.add_figure(title, fig, epoch)
        writer.add_scalar('test/mca', mca, epoch)
        writer.add_scalar('test/acc', acc, epoch)
        writer.add_scalar('test/mcr', mcr, epoch)
        writer.add_scalar('test/mean_f1',m_f1,epoch)
        for lbl,precision_ in enumerate(precision):
            writer.add_scalar('test/class_precision_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),precision_,epoch)
        for lbl,recall_ in enumerate(recall):
            writer.add_scalar('test/class_recall_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),recall_,epoch)
        for lbl,f1_ in enumerate(f1):
            writer.add_scalar('test/class_f1_of_CLASS{}'.format(Name_dict[dataset_name][lbl]),f1_,epoch)

    return val_loss, metrics , m_f1 , mca , mcr 


def getMetrics(name,correct, predicted):
    acc = accuracy_score(correct,predicted)
    precision = precision_score(correct,predicted,average=None)
    recall = recall_score(correct,predicted,average=None)
    f1 = f1_score(correct,predicted,average=None)
    mca = precision.mean()
    mcr = recall.mean()
    m_f1 = f1.mean()
    return acc, precision, recall, f1, mca, mcr, m_f1

def plot_confusion_matrix(name, y_true, y_pred, normalized=True):
    classes = Name_dict[name]
    n_classes = len(classes)
    cm = confusion_matrix(y_true,y_pred)
    title = 'confusion matrix {}'
    if normalized == True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title.format('Normalize')
    else:
        title.format('Not Normalize')

    np.set_printoptions(precision=2)
    fig = plt.figure(figsize=(n_classes, n_classes), dpi=320, facecolor='w', edgecolor='k')

    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')


    tick_marks = np.arange(len(classes))
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'f') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    return fig, title