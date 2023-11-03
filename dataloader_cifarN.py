import os
import torch
import copy
import random
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.preprocessing import normalize
import math
from scipy import integrate


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def initial_data(dataset, noise_type, noise_path, root_dir):
    def load_label(noise_path, noise_type):
        print(noise_path, noise_type)
        noise_label = torch.load(noise_path)
        print(noise_label[noise_type].reshape(-1).shape)
        return noise_label[noise_type].reshape(-1)

    print('============ Initialize data')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if dataset == 'cifar10':
        test_dic = unpickle('%s/test_batch' % root_dir)
        test_data = test_dic['data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic['labels']

        for n in range(1, 6):
            dpath = '%s/data_batch_%d' % (root_dir, n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            train_label = train_label + data_dic['labels']
        train_data = np.concatenate(train_data)
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))

    train_noisy_labels = load_label(noise_path, noise_type)
    train_noisy_labels = train_noisy_labels.tolist()
    print(f'noisy labels loaded from {noise_path}')

    noise_label = train_noisy_labels
    return train_data, train_label, noise_label, test_data, test_label



class cifarn_dataset(Dataset):
    def __init__(self, data, label, transform, mode, pred=[], probability=[]):
        self.transform = transform
        self.mode = mode
        self.pred = pred
        self.probability = None

        if self.mode == 'all' or self.mode == 'test':
            self.data = data
            self.label = label
        else:
            if self.mode == 'labeled':
                pred_idx = self.pred.nonzero()[0]
                self.probability = [probability[i] for i in pred_idx]

            elif self.mode == 'unlabeled':
                pred_idx = (1 - pred).nonzero()[0]

            self.data = data[pred_idx]
            self.label = [label[i] for i in pred_idx]

        self.nb_classes = 10

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1, target

        elif self.mode == 'all':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index

        elif self.mode == 'test':
            img, target = self.data[index], self.label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.data)



class cifarn_dataloader():
    def __init__(self, dataset, noise_type, noise_path, batch_size, num_workers, root_dir, args, is_human=True, noise_file='', noise_mode='cifarn'):
        self.noise_mode = noise_mode
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.is_human = is_human
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.args = args

        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])

        # dataset, noise_type, noise_path, root_dir
        self.train_data, self.train_label, self.noise_label, self.test_data, self.test_label \
            = initial_data(self.dataset, self.noise_type, self.noise_path, self.root_dir)

    def mapping_func(self, w):
        beta = self.args.beta
        def beta_f(t):
            return math.pow(1 - t, beta - 1)

        v, _ = integrate.quad(beta_f, 0, 1)
        tilde_w = 1 / v * math.pow(1 - w, beta - 1)
        return tilde_w


    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifarn_dataset(self.train_data, self.noise_label, transform=self.transform_train, mode="all")
            trainloader = DataLoader(dataset=all_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifarn_dataset(self.train_data, self.noise_label,transform=self.transform_train, mode="labeled", pred=pred, probability=prob)

            data_num = len(labeled_dataset.data)  # a list with size of arg.num_class
            labels = labeled_dataset.label

            # calculate per class number
            cls_number = [0] * 10
            for i in range(data_num):
                cls_number[labels[i]] += 1

            # class number normalization, get v
            cls_number = np.array(cls_number, dtype='float32').reshape(1, -1)
            v = normalize(cls_number, norm='l1').tolist()[0]  # v vector in the paper

            # get tilde_v by mapping function
            tilde_v = [0] * 10
            for i in range(10):
                tilde_v[i] = self.mapping_func(v[i])

            # customize the sampler
            weight_ = [0] * data_num  # assign weight to each training sample
            weight_reversed = [0] * data_num
            for i in range(data_num):
                weight_[i] = v[labels[i]]
                weight_reversed[i] = tilde_v[labels[i]]

            sampler_v = WeightedRandomSampler(weight_, num_samples=data_num, replacement=False)
            sampler_v_tilde = WeightedRandomSampler(weight_reversed, num_samples=data_num, replacement=True)

            labeled_trainloader_v = DataLoader(labeled_dataset, batch_size=self.batch_size, sampler=sampler_v, shuffle=False, num_workers=4, drop_last=True)
            labeled_trainloader_v_tilde = DataLoader(labeled_dataset, batch_size=self.batch_size, sampler=sampler_v_tilde, shuffle=False, num_workers=4, drop_last=True)

            return labeled_trainloader_v, labeled_trainloader_v_tilde

        elif mode == 'test':
            test_dataset = cifarn_dataset(self.test_data, self.test_label,  transform=self.transform_test, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifarn_dataset(self.train_data, self.noise_label,  transform=self.transform_test, mode='all')
            eval_loader = DataLoader(eval_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
            return eval_loader
