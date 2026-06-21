import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F

class RecorderMeter(object):
    def __init__(self, *args, **kwargs):
        pass

import torch
import torch.nn as nn
import math


import torch
import cv2
import numpy as np
import random

import pickle
from torch.autograd import Variable

import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms

import sys
sys.path.append('../')

import clip
# device and clip_model will be handled inside Model and main

from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class my_MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class my_AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()
        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'





class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.phase = phase
        self.transform = transform
        
        if phase == 'train':
            label_file = args.train_label
        else:
            label_file = args.test_label
            
        label_path = os.path.join(args.raf_path, label_file)
        df = pd.read_csv(label_path, sep=' ', header=None)
        
        self.file_paths = [os.path.join(args.raf_path, x) for x in df[0].values]
        self.label = df[1].values
        self.aug_func = [flip_image, add_g]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.phase == 'train' and random.uniform(0, 1) > 0.5:
            image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        
        image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx, image1
    
    

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x
    

    
class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
                
        
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2,2,2,2], channels=[64, 128, 256, 512], stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
import random
import torch
import numpy as np
from torch.autograd import Variable


##### channel dropping 
def Mask(nb_batch):
    bar = []
    for i in range(7):
        foo = [1] * 58 + [0] *  15
        if i == 6:
            foo = [1] * 59 + [0] *  15
        random.shuffle(foo)  #### generate mask
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,512,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

###### channel separation and channel diverse loss
def supervisor(x, targets, cnum):
    branch = x
    branch = branch.reshape(branch.size(0),branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)  
    branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0
    
    mask = Mask(x.size(0))
    branch_1 = x.reshape(x.size(0),x.size(1), 1, 1) * mask 
    branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)  
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
    return [loss_1, loss_2] 
    
    

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0):
        super(Model, self).__init__()
        
        # Load CLIP model inside to facilitate DataParallel
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu') 
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Use ResNet-50 as backbone as specified by the author
        res50 = models.resnet50(pretrained=pretrained)
        msceleb_path = os.path.join(args.raf_path, '/kaggle/input/datasets/nguynquclc/resnet-50/resnet50_pretrained_on_msceleb.pth')
        if os.path.isfile(msceleb_path):
            msceleb_model = torch.load(msceleb_path, map_location='cpu', weights_only=False)
            state_dict = msceleb_model['state_dict'] if isinstance(msceleb_model, dict) and 'state_dict' in msceleb_model else msceleb_model
            res50.load_state_dict(state_dict, strict=False)
            print('Loaded pretrained backbone from {}'.format(msceleb_path))
        else:
            print('Warning: pretrained checkpoint not found at {}. Using ResNet-50 pretrained on ImageNet.'.format(msceleb_path))
        
        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res50.children())[:-2])
        self.features2 = nn.Sequential(*list(res50.children())[-2:-1])
        
        # Reduction layer: reduce 2048 to 512 through mean operation using sliding windows
        self.reduction = nn.AvgPool1d(kernel_size=4, stride=4)
        
        self.fc = nn.Linear(512, num_classes) # Reduced dimension is 512
        
        self.parm={}
        for name,parameters in self.fc.named_parameters():
            print(name,':',parameters.size())
            self.parm[name]=parameters
        
    def forward(self, x, targets, phase='train'):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
            
        x = self.features(x)
        feat = x
        
        x = self.features2(x)
        x = x.view(x.size(0), -1)    
        
        # Reduce dimension from 2048 to 512 using sliding window mean (non-overlapping)
        x = x.view(x.size(0), 1, -1) # (N, 1, 2048)
        x = self.reduction(x) # (N, 1, 512)
        x = x.view(x.size(0), -1) # (N, 512)
        ################### sigmoid mask (important)
        if phase=='train':
            MC_loss = supervisor(image_features * torch.sigmoid(x), targets, cnum=73)

        x = image_features * torch.sigmoid(x)
        out = self.fc(x)
        
        if phase=='train':
            return out, MC_loss
        else:
            return out, out
    
    
    

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

    

import torch.nn.functional as F
from torch.autograd import Variable


    

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='/kaggle/input/datasets/nguynquclc/ferplus/FERPlus', help='dataset_path')
parser.add_argument('--train_label', type=str, default='train_ferplus.txt', help='train_label_file')
parser.add_argument('--test_label', type=str, default='test_ferplus.txt', help='test_label_file')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
args = parser.parse_args(args=[])





def train(args, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.to(device)
    model.train()
    pbar = tqdm(train_loader, desc="Training")
    
    total_loss = []
    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(pbar):
        imgs1 = imgs1.to(device)
        labels = labels.to(device)
        
    
        criterion = nn.CrossEntropyLoss(reduction='none')

        

        output, MC_loss = model(imgs1, labels, phase='train')
        
        loss1 = nn.CrossEntropyLoss()(output, labels)



        loss = loss1 + 4.0 * MC_loss[1].mean() + 2.0 * MC_loss[0].mean() 


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num.item()
        running_loss += loss.item()
        pbar.set_postfix(Loss=f"{loss.item():.4f}", Acc=f"{float(correct_sum)/(iter_cnt*args.batch_size):.2f}")

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = float(correct_sum) / float(train_loader.dataset.__len__())
    return acc, running_loss


    
def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0


        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)


            outputs, _ = model(imgs1, labels, phase='test')


            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num.item()

            running_loss += loss.item()
            data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = float(correct_sum) / float(data_num)
        
    return test_acc, running_loss
        
        
        
def main():    
    setup_seed(3407)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    

    train_dataset = RafDataset(args, phase='train', transform=train_transforms)
    test_dataset = RafDataset(args, phase='test', transform=eval_transforms)
    


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=False)
    
    
    
    
    model = Model()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters() , lr=0.0002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
        test_acc, test_loss = test(model, test_loader, device)
        print('epoch: ', i, 'acc: ', test_acc)
        if test_acc>best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(),}, "ours_best.pth") 
        torch.save({'model_state_dict': model.state_dict(),}, "ours_final.pth") 
        with open('results.txt', 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'\n')



if __name__ == '__main__':
    main()