import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose
from PIL.Image import BICUBIC
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau

import random
import os
import argparse
from itertools import chain

from densenet import *
from utils import progress_bar, count_parameters
from efficientnet import EfficientNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='DenseNetDeep', type=str, help='model type')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decay', default=0.97, type=float, help='learning rate decay')
parser.add_argument('--decay_epoch', default=3, type=int, help='learning rate decay epoch')
parser.add_argument('--decay_mode', default='StepLR', type=str, help='learning rate decay mode')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
parser.add_argument('--save_name', default='', type=str, help='save file name')
parser.add_argument('--seed', default=51, type=int, help='random seed')
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
checkpoint_dir = './checkpoint/' + args.model + args.save_name
model = args.model
fine_tune = True if 'EfficientNet' in model else False

# Data
print('==> Preparing data..')
if fine_tune:
    image_size = 224 if model == 'EfficientNetB0' else 300
    transform_train = Compose([
        transforms.Resize(int(image_size * 1.1), BICUBIC),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124, 117, 104)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = Compose([
        transforms.Resize(image_size, BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    transform_train = transforms.Compose([
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
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if model == 'DenseNetWide': net = DenseNet(depth=106, k=13, num_classes=10) # 992,841
if model == 'DenseNetDeep': net = DenseNet(depth=117, k=12, num_classes=10) # 932,986
if model == 'EfficientNetB0': net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10) # 4,020,358
if model == 'EfficientNetB4': net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10) # 17,566,546
print('Number of parameters: ', count_parameters(net))
net = net.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
if fine_tune:
    lr = args.lr
    optimizer = optim.SGD([
        {
            "params": chain(net._conv_stem.parameters(), net._bn0.parameters(), net._blocks.parameters()),
            "lr": lr * 0.1,
        },
        {
            "params": chain(net._conv_head.parameters(), net._bn1.parameters()),
            "lr": lr * 0.5,
        },
        {
            "params": net._fc.parameters(),
            "lr": lr
        }],
        momentum=0.9, weight_decay=1e-4, nesterov=True)
        # When I was training EfficientNet-B4, I mistakenly set the weight_decay as 1e-3 and did not try 1e-4.
    # lr_scheduler = ExponentialLR(optimizer, gamma=args.decay)
    if args.decay_mode == 'StepLR':
        lr_scheduler = StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay)
    else:
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.decay, patience=5)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Losses file
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
log_file_name = os.path.join(checkpoint_dir, 'log.txt')
log_file = open(log_file_name, "at")


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_dir + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if (batch_idx % 90 == 0):
            log = 'Epoch: {}, Step: {}, Train loss: {:.4f}, Training acc: {:.4f}\n'.format(epoch, batch_idx, train_loss/(batch_idx+1), correct/total)
            log_file.write(log)
            log_file.flush()

    return train_loss/(batch_idx+1)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

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
            'net': net.state_dict(),
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
    if fine_tune:
        if args.decay_mode == 'StepLR':
            lr_scheduler.step()
        else:
            lr_scheduler.step(train_loss)
    else:
        adjust_learning_rate(optimizer, epoch)
    test(epoch)
