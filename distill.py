import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, RandomAffine, ToTensor, Normalize
from PIL.Image import BICUBIC
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau

import random
import os
import argparse
from cifar_data import *
from itertools import chain

from densenet import *
from utils import progress_bar, count_parameters
from efficientnet import EfficientNet

parser = argparse.ArgumentParser(description='CIFAR10 Distillation')
parser.add_argument('--model', default='DenseNetDeep', type=str, help='model type')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--coef', default=0.2, type=float, help='coefficient in kd objective')
# parser.add_argument('--coef2', default=1, type=float, help='coefficient in kd objective')
parser.add_argument('--temperature', default=10, type=float, help='temperature in softmax')
# parser.add_argument('--decay', default=0.975, type=float, help='learning rate decay')
# parser.add_argument('--decay_epoch', default=1, type=int, help='learning rate decay epoch')
# parser.add_argument('--decay_mode', default='StepLR', type=str, help='learning rate decay mode')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pre_train', action='store_true', help='load a pre-trained small model')
parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
parser.add_argument('--save_name', default='', type=str, help='save file name')
parser.add_argument('--seed', default=51, type=int, help='random seed')
args = parser.parse_args()


random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
checkpoint_dir_big = './checkpoint/EfficientNetB4lr1e-2decay.5per30/ckpt.pth'
checkpoint_dir = './checkpoint/Distill' + args.model + '_' + args.save_name
checkpoint_dir_small = './checkpoint/DenseNetWideseed223'


image_size = 300
transform_big = Compose([
    # transforms.ToPILImage(mode='RGB'),
    Resize(300, BICUBIC),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_small = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_dir = '~/data'
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
trainset = MultiTransDataset(trainset, transform_big, transform_small)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net_big = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10).to(device)
net_big = torch.nn.DataParallel(net_big)
cudnn.benchmark = True

checkpoint = torch.load(checkpoint_dir_big)
net_big.load_state_dict(checkpoint['net'])
net_big.eval()


if args.model == 'DenseNetWide': net_small = DenseNet(depth=106, k=13, num_classes=10).to(device) # <1m
if args.model == 'DenseNetDeep': net_small = DenseNet(depth=117, k=12, num_classes=10).to(device) # <1m

if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
log_file_name = os.path.join(checkpoint_dir, 'log.txt')
log_file = open(log_file_name, "at")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_dir + '/ckpt.pth')
    net_small.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

net_small = torch.nn.DataParallel(net_small)
if not args.resume and args.pre_train:
    print('==> Load a pretrained small model..')
    assert os.path.isdir(checkpoint_dir_small), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_dir_small + '/ckpt.pth')
    net_small.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']

# Loss function for knowledge distillation
def kd_loss(true_label, small_logits, big_logits, coef=1, temperature=1):
    obj_correct = F.cross_entropy(small_logits, true_label)

    # big_soft = F.softmax(big_logits / temperature, dim=1)
    # small_soft = F.softmax(small_logits / temperature, dim=1)
    # obj_big_small = torch.mean(torch.sum(- big_soft * torch.log(small_soft), dim=1))
    obj_big_small = F.kl_div(F.log_softmax(small_logits/temperature, dim=1), F.softmax(big_logits/temperature, dim=1), reduction='batchmean')

    return coef * obj_correct + (1 - coef) * temperature**2 * obj_big_small, obj_correct, obj_big_small


optimizer = optim.SGD(net_small.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) #1e-4

# lr_scheduler = StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net_small.train()
    train_loss = 0
    train_loss_ori = 0
    train_loss_teach = 0
    correct = 0
    total = 0
    for batch_idx, (inputs_big, inputs_small, targets) in enumerate(trainloader):
        # inputs_small = torch.stack(list(map(transform_small, inputs)))
        # inputs_big = torch.stack(list(map(transform_big, inputs)))
        inputs_small, inputs_big, targets = inputs_small.to(device), inputs_big.to(device), targets.to(device)
        optimizer.zero_grad()
        # inputs_small = F.adaptive_avg_pool2d(inputs, output_size=32)
        outputs = net_small(inputs_small)
        with torch.no_grad():
            big_outputs = net_big(inputs_big)
        loss, loss_ori, loss_teach = kd_loss(true_label=targets, small_logits=outputs, big_logits=big_outputs,
                       coef=args.coef, temperature=args.temperature)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_ori += loss_ori.item()
        train_loss_teach += loss_teach.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if (batch_idx % 90 == 0):
            log = 'Epoch: {}, Step: {}, Train loss: {:.4f}, Original loss: {:.4f}, Teaching loss: {:.4f}, Training acc: {:.4f}\n'.format(
                epoch, batch_idx, train_loss/(batch_idx+1), train_loss_ori/(batch_idx+1), train_loss_teach/(batch_idx+1), correct/total)
            log_file.write(log)
            log_file.flush()

    return train_loss/(batch_idx+1)


def test(epoch):
    global best_acc
    net_small.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net_small(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net_small.module.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_dir + '/ckpt.pth')
        best_acc = acc
    print('Current best testing accuracy: %.3f%%' % best_acc)
    log = 'Epoch: {}, Test loss: {:.4f}, Test acc: {:.4f}, Current best: {:.4f}\n'.format(
        epoch, test_loss / (batch_idx + 1), correct / total, best_acc / 100)
    log_file.write(log)
    log_file.flush()


def adjust_learning_rate(optimizer, epoch):
    if epoch < 150:
        lr = args.lr
    elif epoch < 225:
        lr = args.lr * 0.1
    elif epoch < 300:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(start_epoch, start_epoch + args.epochs):
    train_loss = train(epoch)
    adjust_learning_rate(optimizer, epoch)
    test(epoch)
