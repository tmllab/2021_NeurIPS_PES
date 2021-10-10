import os
import os.path
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.datasets import CIFAR10, CIFAR100

from networks.ResNet import ResNet18, ResNet34
from common.tools import getTime, evaluate, predict_softmax, train
from common.NoisyUtil import Train_Dataset, dataset_split, Semi_Unlabeled_Dataset


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
parser.add_argument('--num_epochs', default=200, type=int)

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', type=str, default='./data', help='data directory')
parser.add_argument('--data_percent', default=0.9, type=float, help='data number percent')
parser.add_argument('--noise_type', default='symmetric', type=str)
parser.add_argument('--noise_rate', default=0.5, type=float, help='corruption rate, should be less than 1')
parser.add_argument('--model_name', default='resnet18', type=str)

parser.add_argument('--warmup', default=0, type=int, help='warmup epochs, 0 means default')
parser.add_argument('--refine_lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--refine_times_1', default=7, type=int, help='default 5')
parser.add_argument('--refine_times_2', default=7, type=int, help='default 5')
parser.add_argument('--refine_times_3', default=0, type=int, help='default 5')

args = parser.parse_args()
print(args)
os.system('nvidia-smi')

args.model_dir = 'model/'
if not os.path.exists(args.model_dir):
    os.system('mkdir -p %s' % (args.model_dir))

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True
    cudnn.benchmark = True


def create_model(name="resnet18", input_channel=3, num_classes=10):
    if(name == "resnet18"):
        model = ResNet18(num_classes)
    else:
        print("create ResNet34")
        model = ResNet34(num_classes)

    model.cuda()
    return model


def splite_confident(outs, clean_targets, noisy_targets):
    probs, preds = torch.max(outs.data, 1)
    confident_correct_num = 0
    confident_indexs = []
    for i in range(0, len(noisy_targets)):
        if preds[i] == noisy_targets[i]:
            confident_indexs.append(i)

            if clean_targets[i] == preds[i]:
                confident_correct_num += 1

    print(getTime(), "Confident:", len(confident_indexs), round(confident_correct_num / len(confident_indexs) * 100, 2))
    return confident_indexs


def update_trainloader(model, train_data, clean_targets, noisy_targets, fixed_confident_indexs=None):
    predict_dataset = Semi_Unlabeled_Dataset(train_data, transform_train)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    soft_outs = predict_softmax(predict_loader, model)
    confident_indexs = splite_confident(soft_outs, clean_targets, noisy_targets)
    confident_dataset = Train_Dataset(train_data[confident_indexs], noisy_targets[confident_indexs], transform_train)
    train_loader = DataLoader(dataset=confident_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # Loss function
    train_nums = np.zeros(args.num_class, dtype=int)
    for item in noisy_targets[confident_indexs]:
        train_nums[item] += 1

    with np.errstate(divide='ignore'):
        cw = np.mean(train_nums[train_nums != 0]) / train_nums
        cw[cw == np.inf] = 0

    class_weights = torch.FloatTensor(cw).cuda()
    print("Category", train_nums, "precent", class_weights)
    ceriation = nn.CrossEntropyLoss(weight=class_weights).cuda()
    return train_loader, ceriation


def noisy_refine(model, train_loader, num_layer, refine_times):
    if refine_times <= 0:
        return model

    update_trainloader(model, train_data, train_clean_labels, train_noisy_labels)
    # frezon all layers and add a new final layer
    for param in model.parameters():
        param.requires_grad = False

    model.renew_layers(num_layer)
    model.cuda()

    optimizer_refine = torch.optim.Adam(model.parameters(), lr=args.refine_lr)
    for epoch in range(refine_times):
        train(model, train_loader, optimizer_refine, ceriation, epoch)
        _, test_acc = evaluate(model, test_loader, ceriation, "Refine:" + str(epoch))

    for param in model.parameters():
        param.requires_grad = True

    return model


if args.dataset == 'cifar10' or args.dataset == 'CIFAR10':
    if args.warmup == 0:
        args.warmup = 25
    args.num_class = 10
    args.model_name = "resnet18"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_set = CIFAR10(root=args.data_path, train=True, download=True)
    test_set = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100' or args.dataset == 'CIFAR100':
    if args.warmup == 0:
        args.warmup = 30
    args.num_class = 100
    args.model_name = "resnet34"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_set = CIFAR100(root=args.data_path, train=True, download=True)
    test_set = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)

train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, _ = dataset_split(train_set.data, np.array(train_set.targets), args.noise_rate, args.noise_type, args.data_percent, args.seed, args.num_class, False)
train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_train)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8, pin_memory=True)

model = create_model(name=args.model_name, num_classes=args.num_class)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
ceriation = nn.CrossEntropyLoss().cuda()
train_ceriation = ceriation

best_val_acc = 0
best_test_acc = 0
for epoch in range(args.num_epochs):
    if epoch < args.warmup:
        train(model, train_loader, optimizer, train_ceriation, epoch)
    elif epoch == args.warmup:
        model = noisy_refine(model, train_loader, 2, args.refine_times_3)
        model = noisy_refine(model, train_loader, 1, args.refine_times_2)
        model = noisy_refine(model, train_loader, 0, args.refine_times_1)
    else:
        train_loader, train_ceriation = update_trainloader(model, train_data, train_clean_labels, train_noisy_labels)
        train(model, train_loader, optimizer, train_ceriation, epoch)

    scheduler.step()
    _, val_acc = evaluate(model, val_loader, ceriation, "Val Acc:")
    if best_val_acc < val_acc:
        _, test_acc = evaluate(model, test_loader, ceriation, "Epoch " + str(epoch) + " Test Acc:")
        best_test_acc = test_acc
        best_val_acc = val_acc

print(getTime(), "Best Test Acc:", best_test_acc)