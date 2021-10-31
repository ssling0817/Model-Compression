import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
import os
import copy
import time
from res_model import ResNet50
from torch.nn.utils import prune
import thop
from thop import profile
from pruning_layers import MaskedLinear, MaskedConv2d 
import numpy as np

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix

def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc

def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    '''    
    all_weights = []
    for p in model.modules():
        #print(p.state_dict())
        if isinstance(p, MaskedLinear) or isinstance(p, MaskedConv2d) :
            print(p)
            sd = p.state_dict()
            for k in sd.keys():
                if not 'weight' in k:
                    continue
                w = sd[k]
            if len(w.size()) != 1:#w = p.weight.data
                all_weights += list(w.cpu().data.abs().numpy().flatten())
        if isinstance(p, nn.Sequential):
            #print(type(p))
            for blocks in p:
                
                if isinstance(blocks, MaskedLinear) or isinstance(blocks, MaskedConv2d) :
                   # print(blocks)
                    sd = blocks.state_dict()
                    for k in sd.keys():
                        if not 'weight' in k:
                            continue
                        w = sd[k]
                    if len(w.size()) != 1:#w = p.weight.data
                        all_weights += list(w.cpu().data.abs().numpy().flatten())

  #  print('end')
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    masks = []
    for p in model.modules():
        if isinstance(p, MaskedLinear) or isinstance(p, MaskedConv2d) :
            sd = p.state_dict()
            for k in sd.keys():
                if not 'weight' in k:
                    continue
                w = sd[k]
            if len(w.size()) != 1:
                pruned_inds = w.abs() > threshold
                masks.append(pruned_inds.float())
        if isinstance(p, nn.Sequential):
            #print(p)
            for blocks in p:
                if isinstance(blocks, MaskedLinear) or isinstance(blocks, MaskedConv2d) :
                    sd = blocks.state_dict()
                    for k in sd.keys():
                        if not 'weight' in k:
                            continue
                        w = sd[k]
                    if len(w.size()) != 1:
                        pruned_inds = w.abs() > threshold
                        masks.append(pruned_inds.float())
    return masks
def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    '''
    NO_MASKS = False
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for module in model.modules():
        if isinstance(module, MaskedConv2d):
            sd = module.state_dict()
            for k in sd.keys():
                if not 'weight' in k:
                    continue
                w = sd[k]
            p_np = w.cpu().numpy()
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks
    '''

    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks
    '''

def filter_prune(model, pruning_perc):
    masks = []
    current_pruning_perc = 0.
 
    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        ind=0
        for module in model.modules():
            if isinstance(module, MaskedConv2d):
                if type(masks[ind])==np.ndarray:
                    masks[ind] = torch.from_numpy(masks[ind])
                module.set_mask(masks[ind])
                ind = ind + 1
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks

def test():
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))
            if GPU:
                image = image.cuda()
                label = label.cuda()
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset),' ',"Top 1 acc: ", correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset),' ',"Top 5 acc: ", correct_5 / len(cifar100_test_loader.dataset))
#00 add
    return correct_1 / len(cifar100_test_loader.dataset)
    #print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

if __name__ == '__main__':
    GPU = True
    WEIGHTS_FILE = '5_w_9_f_0.8036.pth'
    #args = parse_args()
    #args.num_class = 100
    #args.Use_Cuda = torch.cuda.is_available()
    #args.sum_channel = None

    BATCH_SIZE = 128
    
    net = ResNet50()
    if GPU:
        net = net.cuda()
    '''
    masks = filter_prune(net, 0.01)
    ind=0
    for module in net.modules():
        if isinstance(module, MaskedConv2d):
            module.set_mask(masks[ind])
            ind = ind + 1
    '''
    #< prepare net >

    masks = weight_prune(net, 0)
    ind=0
    #print(type(masks))
    #net.set_masks(masks)
    
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
    
    net.load_state_dict(torch.load(WEIGHTS_FILE))
    

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=True, num_workers=1, batch_size=BATCH_SIZE)

    net.eval()
    test()
    
    papameter_sum_b = 0
    sparse_b = 0
    for k,v in net.named_parameters():
        papameter_sum_b += (torch.sum(v!=0)).item()
        sparse_b += (torch.sum(v==0)).item()
   # for k,v in net.state_dict().items():
   #     print(k)
    print("-----Before Pruning-----")
    print("papameters_0",sparse_b)
    print("tatal",papameter_sum_b+sparse_b)
    print("ratio",sparse_b/(papameter_sum_b+sparse_b))

    '''
    #< filter pruning part >
    masks = filter_prune(net, 17.5)
    print(len(masks))
    ind=0
    for module in net.modules():
        if isinstance(module, MaskedConv2d):
            module.set_mask(masks[ind])
            ind = ind + 1
    '''
    #< weight pruning part >
    masks = weight_prune(net, 74.7)
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
    
    # for k,v in net.state_dict().items():
    #     print(k)
    torch.save(net.state_dict(), '5_f_10_w.pth')
    print('saving weights file to 5_f_10_w.pth') 
#00 add
    correct = test()
    papameter_sum = 0
    sparse = 0
    for k,v in net.named_parameters():
        #print('v',v)
        papameter_sum += (torch.sum(v!=0)).item()
        sparse += (torch.sum(v==0)).item()
    print("-----After Pruning-----")
    print("papameters_0",sparse)
    print("total",papameter_sum+sparse)
    print("ratio",sparse/(papameter_sum+sparse))
	
   
    
    
