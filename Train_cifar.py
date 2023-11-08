from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
import time

from my_ResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='ITEM')
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--noise_mode', default='sym', help='sym, instance')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')

parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--seed', default=42)
parser.add_argument('--gpuid', default=0, type=int)

parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--cls_num', default=3, type=int)
parser.add_argument('--mixup', default='mixup', type=str, help='normal, mixup')
parser.add_argument('--beta', default=3, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def pprint(content, txtfile):
    nowTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(nowTime + ' | ' + content)
    with open(txtfile, "a") as myfile:
        myfile.write(nowTime + ' | ' + content + '\n')


# Training
def train(epoch, net, optimizer, trainloader_v, trainloader_v_tilde):
    net.train()

    train_iter_v = iter(trainloader_v)
    train_iter_v_tilde = iter(trainloader_v_tilde)
    num_iter = (len(trainloader_v.dataset) // args.batch_size) + 1

    calculate_num = [0] * args.cls_num
    for batch_idx in range(num_iter * args.cls_num):
        try:
            inputs_x1, labels_x1 = next(train_iter_v)
        except:
            train_iter_v = iter(trainloader_v)
            inputs_x1, labels_x1 = next(train_iter_v)

        try:
            inputs_x2, labels_x2 = next(train_iter_v_tilde)
        except:
            train_iter_v_tilde = iter(trainloader_v_tilde)
            inputs_x2, labels_x2 = next(train_iter_v_tilde)

        batch_size = inputs_x1.size()[0]
        labels_x1 = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x1.view(-1, 1), 1)
        labels_x2 = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x2.view(-1, 1), 1)

        inputs_x1, labels_x1 = inputs_x1.cuda(), labels_x1.cuda()
        inputs_x2, labels_x2 = inputs_x2.cuda(), labels_x2.cuda()

        selected_cls_idx = random.randint(0, args.cls_num - 1)
        calculate_num[selected_cls_idx] += 1

        if args.mixup == 'normal':
            outputs_x = net(inputs_x)[selected_cls_idx]
            train_loss = CEloss(outputs_x, labels_x)

        elif args.mixup == 'mixup':
            l = np.random.beta(1, 1) # Uniform sampling from [0,1]
            l = max(l, 1 - l)

            mixed_input = l * inputs_x1 + (1 - l) * inputs_x2
            mixed_target = l * labels_x1 + (1 - l) * labels_x2
            mixed_outputs = net(mixed_input)[selected_cls_idx]

            train_loss = -torch.mean(torch.sum(F.log_softmax(mixed_outputs, dim=1) * mixed_target, dim=1))

        else:
            ValueError('Error training strategy .......')

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            pprint('%s:%.1f-%s | Training | Iter[%3d/%3d]\t  Labeled loss: %.2f\t'
                % (args.dataset, args.r, args.noise_mode, batch_idx + 1, num_iter * args.cls_num, train_loss.item()), text_name)

    pprint('Selected numbers on different experts: ' + str(calculate_num), text_name)


def warmup(epoch, net, optimizer, dataloader):
    net.train()

    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    calculate_num = [0] * args.cls_num
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        all_layer_outputs = net(inputs)

        cls_index = random.randint(0, args.cls_num - 1)
        calculate_num[cls_index] += 1

        outputs = all_layer_outputs[cls_index]
        loss = CEloss(outputs, labels)
        if args.noise_mode == 'pair':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode == 'sym' or 'instance':
            L = loss
        L.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            pprint('%s:%.1f-%s | WarmUp | Iter[%3d/%3d]\t CE-loss: %.4f\t' % (args.dataset, args.r,
                                    args.noise_mode, batch_idx + 1, num_iter, loss.item()), text_name)
    pprint('Selected numbers on different experts: ' + str(calculate_num), text_name)


def test(epoch, net):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_list = net(inputs)
            outputs1 = sum([output for output in outputs_list]) / args.cls_num

            _, predicted = torch.max(outputs1, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    pprint("Test Accuracy on epoch [%3d/%3d]: %.2f%%\n" % (epoch, args.num_epochs, acc), text_name)
    return acc


# ensemble all clas_layer
def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            bs = inputs.size(0)
            outputs_list = model(inputs)

            temp_tensor = torch.zeros(bs).cuda()
            for i in range(args.cls_num):
                temp_tensor += CE(outputs_list[i], targets)
            avg_loss = temp_tensor / args.cls_num

            for b in range(bs):
                losses[index[b]] = avg_loss[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)
    input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model(name, cls_layer_num):
    if name == 'cifar10':
        model = ResNet18(layer_num=cls_layer_num, num_classes=args.num_class)
        pprint('Building %s net for %s' % ("ResNet-18", name), text_name)
    elif name == 'cifar100':
        model = ResNet34(layer_num=cls_layer_num, num_classes=args.num_class)
        pprint('Building %s net for %s' % ("ResNet-34", name), text_name)
    else:
        model = None

    model = model.cuda()
    return model

# ----------------------------------------------- Main ------------------------------------------------
Time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
text_name = './Log2/%s(%s_%.1f)_beta(%.1f)_ClsNum(%.1f)_%s' % (args.dataset, args.noise_mode, args.r,
                                                              args.beta, args.cls_num, args.mixup)
text_name = text_name + '_' + Time + '_acc.txt'

# '--------args----------'
dict_args = {}
for k in list(vars(args).keys()):
    dict_args[k] = vars(args)[k]
pprint(str(dict_args), text_name)


if args.dataset == 'cifar10':
    args.warm_up = 10
    args.num_class = 10
    args.data_path = './dataset/cifar-10'
elif args.dataset == 'cifar100':
    args.warm_up = 30
    args.num_class = 100
    args.data_path = './dataset/cifar-100'
else:
    args.warm_up = 1


loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=8, root_dir=args.data_path, args=args,
                                         noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

net1 = create_model(name=args.dataset, cls_layer_num=args.cls_num)
pprint("Set classifier layer number = {}, total params = {:.2f}M \n".format(args.cls_num,
                                                sum(p.numel() for p in net1.parameters()) / 1e6), text_name)

cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[100, 150], gamma=0.1)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[]]  # save the history of losses from two networks
best_acc = 0

for epoch in range(args.num_epochs + 1):
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch < args.warm_up:
        warmup_trainloader = loader.run('warmup')
        pprint('Epoch {}'.format(epoch), text_name)
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        testacc = test(epoch, net1)

    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        pred1 = (prob1 > 0.5)

        pprint('Train ITEM', text_name)
        trainloader_v, trainloader_v_tilde = loader.run('train', pred1, prob1)  # co-divide
        train(epoch, net1, optimizer1, trainloader_v, trainloader_v_tilde)  # train net1
        testacc = test(epoch, net1)

    lr_schedule.step()
    if best_acc < testacc:
        best_acc = testacc

pprint("Best Accuracy: %.2f%%\n" % (best_acc), text_name)
