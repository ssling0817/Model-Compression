import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import os
import time
from res_model import ResNet50
from pruning import weight_prune, filter_prune
from pruning_layers import MaskedLinear, MaskedConv2d 
#from res_model import ResNetBottleNeckBlock
#def resnet50(in_channels=3, n_classes=100, block=ResNetBottleNeckBlock, *args, **kwargs):
 #   return ResNet50(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)
def train(epoch,loss_function):

    start = time.time()
    net.train()
    loss_avg=0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        #print(images.size())

        if GPU:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()*images.size(0)

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1
#00 add trained_samples
        trained_samples=batch_index * BATCH_SIZE + len(images)
        if(trained_samples%10240==0):
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.7f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * BATCH_SIZE + len(images),
                total_samples=len(cifar100_training_loader.dataset)
            ))
        if writer != None:
            writer.add_scalar('Train/loss', loss.item(), n_iter)
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return loss_avg / len(cifar100_training_loader.dataset)


def eval_training(epoch=0, writer=None):

    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if GPU:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if writer:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)
net = ResNet50()
net = net.cuda()
#<prepare filter prune >
'''
masks = filter_prune(net, 0.01)
ind=0
for module in net.modules():
    if isinstance(module, MaskedConv2d):
        module.set_mask(masks[ind])
        ind = ind + 1
'''
masks = weight_prune(net, 50)
ind=0
for module in net.modules():
    if isinstance(module, MaskedLinear) or isinstance(module, MaskedConv2d):
       # print(module)
        module.set_mask(masks[ind])
        ind = ind + 1
    if isinstance(module, nn.Sequential):
       # print(module)
        for blocks in module:
            if isinstance(blocks, MaskedLinear) or isinstance(blocks, MaskedConv2d) :
               # print(blocks)
                blocks.set_mask(masks[ind])
                ind = ind + 1

#state_dict = torch.load('1.pth') 
#net.load_state_dict({k.replace('module.',''):v for k,v in torch.load('1.pth').items()})
net.load_state_dict(torch.load("5_f_10_w_0.8009.pth"))#,strict=False)

if __name__ == '__main__':

    BATCH_SIZE = 256
    LR = 0.0001
    EPOCH = 200
    GPU = True
    weights_path = ''
    
    
    

    #data preprocessing:
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=True, num_workers=1, batch_size=BATCH_SIZE)
    
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=True, num_workers=1, batch_size=BATCH_SIZE)
    
    loss_function = nn.CrossEntropyLoss()
    
    
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= [60, 120, 160, 200], gamma=0.2)
    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=5,min_lr =0.0000001, verbose=True)
   
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    checkpoint_path = os.path.join('model_saved', 'resnet50')#, datetime.now().strftime(DATE_FORMAT))

    if not os.path.exists('runs'):
        os.mkdir('runs')

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join('runs', 'resnet50'))#, datetime.now().strftime(DATE_FORMAT)))
    except:
        pass

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}-{type}.pth')

    best_acc = 0.7
    
    for epoch in range(1, EPOCH + 1):
    #00 add loss_function    
        loss = train(epoch,loss_function)
        train_scheduler.step(loss)
        
        with torch.no_grad():
            acc = eval_training(epoch,writer)

        if  best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % 100:
            weights_path = checkpoint_path.format(epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    if writer != None:
        writer.close()
