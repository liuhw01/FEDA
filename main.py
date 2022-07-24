import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import datetime
#from util import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(train_loader, model, criterion, optimizer, epoch, kwargs):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for i, (images, target, fn) in enumerate(
            train_loader):  # the first i in index, and the () is the content
        images = images.cuda()
        target = target.cuda()
        # compute output  1:自适应 2：局部 3：全局
        output = model(images)

        loss =  criterion(output, target)
        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 4))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % kwargs['print_freq'] == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, kwargs):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target, fn) in  enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output  1:自适应 2：局部 3：全局
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 4))
            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))
            if i % kwargs['print_freq'] == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
best_acc = 0
print('Training time: ' + now.strftime("%m-%d %H:%M"))

kwargs = {'num_workers': 0,
          'pin_memory': True,
          'print_freq':10,
          'checkpoint_path':'F:\\pydata\\data\\FER\\FERPlus-master_new\\otherdata\\recoder\\' + time_str + 'model.pth',
          'best_checkpoint_path': 'F:\\pydata\\data\\FER\\FERPlus-master_new\\otherdata\\recoder\\' + time_str +'model_best.pth',
          'save_path':'F:\\pydata\\data\\FER\\FERPlus-master_new\\otherdata\\recoder',
          'data_root_train':'E:\\FER data\\FERP\\crop\\data_train', # 新的数据集
          'data_root_val': 'E:\\FER data\\FERP\\crop\\data_test',  # 新的数据集
          'data_label_train':'F:\pydata\data\FER\FERPlus-master_new\otherdata\FERP\data_label_train_new_RAF.txt', #新的数据集label
          'data_label_val':'F:\pydata\data\FER\FERPlus-master_new\otherdata\FERP\data_label_val_new_RAF.txt', #新的数据集label
          'lr':0.01,
          'momentum':0.9,
          'weight_decay':1e-4,
          'epochs': 100,
          'batch_size':20,
          }

save_path=kwargs['save_path']
lr=kwargs['lr']
momentum=kwargs['momentum']
weight_decay=kwargs['weight_decay']
epochs=kwargs['epochs']
batch_size = kwargs['batch_size']

vgg16=models.vgg16(pretrained=True)
vgg16.classifier[6]=nn.Linear(4096,7)
#vgg16.fc=nn.Linear(512,7)

"""
checkpoint_load=r'F:\pydata\data\FER\FERPlus-master_new\otherdata\recoder_RAF\[07-22]-[21-22]-model_best.pth' # 6au+landmark

checkpoint = torch.load(checkpoint_load)
#我们的
model_dict = vgg16.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
nam=pretrained_dict.keys()

model_dict.update(pretrained_dict)
vgg16.load_state_dict(model_dict)


"""

vgg16.cuda()
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(vgg16.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
recorder = RecorderMeter(epochs)



cudnn.benchmark = True
# data_root 数据集  data_label 数据集标签

data_root_train = kwargs['data_root_train']
data_root_test = kwargs['data_root_val']

data_label_train=  kwargs['data_label_train']
data_label_val=  kwargs['data_label_val']

mytransform = transforms.Compose([transforms.Resize(224),

                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomAffine((-15, 15), translate=(0.05, 0.05), scale=(0.9, 1.05),
                                                          fillcolor=0),
                                  transforms.ToTensor()])  # transform [0,255] to [0,1]


mytransform1 = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])  # transform [0,255] to [0,1]


train_label = choose_data2(data_label_train)
test_label = choose_data1(data_label_val)

#train_label, test_label = choose_data(data_label)
train_data=myImageFloder(root=data_root_train, label=train_label, transform=mytransform)
test_data=myImageFloder(root=data_root_test, label=test_label, transform=mytransform1)
val_data=myImageFloder(root=data_root_test, label=test_label, transform=mytransform1)


# 读取训练集
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True,num_workers= kwargs['num_workers'], pin_memory=kwargs['pin_memory'])
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=True,num_workers= kwargs['num_workers'], pin_memory=kwargs['pin_memory'])
val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size, shuffle=True,num_workers= kwargs['num_workers'], pin_memory=kwargs['pin_memory'])



start_epoch=0
for epoch in range(start_epoch, epochs):
    start_time = time.time()
    current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print('Current learning rate: ', current_learning_rate)
    txt_name = 'log\\' + time_str + 'log.txt'
    with open(os.path.join(save_path,txt_name), 'a') as f:
        f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

    # train for one epoch
    train_acc, train_los = train(train_loader, vgg16, criterion, optimizer, epoch, kwargs)

    # evaluate on validation set
    val_acc, val_los = validate(val_loader, vgg16, criterion, kwargs)

    scheduler.step()

    recorder.update(epoch, train_los, train_acc, val_los, val_acc)
    curve_name = time_str + 'cnn.png'
    recorder.plot_curve(os.path.join('./log/', curve_name))

    # remember best acc and save checkpoint
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)

    print('Current best accuracy: ', best_acc.item())
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': vgg16.state_dict(),
                     'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(),
                     'recorder': recorder}, is_best, kwargs)
    end_time = time.time()
    epoch_time = end_time - start_time
    print("An Epoch Time: ", epoch_time)
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write(str(epoch_time) + '\n')





