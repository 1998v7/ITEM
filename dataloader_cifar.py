from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from noise_build import dataset_split
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.preprocessing import normalize
import math
from scipy import integrate

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


def initial_data(dataset, r, noise_mode, root_dir, args):
    print('============ Initialize data')
    num_classes = None
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    if dataset == 'cifar10':
        num_classes = 10
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
    elif dataset =='cifar100':
        num_classes = 100
        test_dic = unpickle('%s/test' % root_dir)
        test_data = test_dic['data']
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))
        test_label = test_dic['fine_labels']

        train_dic = unpickle('%s/train' % root_dir)
        train_data = train_dic['data']
        train_label = train_dic['fine_labels']
    train_data = train_data.reshape((50000, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))

    # build noise
    noise_label = dataset_split(train_images=train_data, train_labels=train_label,
                                noise_rate=r, noise_type=noise_mode,
                                random_seed=args.seed, num_classes=num_classes)

    print('============ Actual clean samples number: ', sum(np.array(noise_label) == np.array(train_label)))
    return train_data, train_label, noise_label, test_data, test_label


class cifar_dataset(Dataset):
    def __init__(self, args, data, real_label, label, transform, mode, strong_transform=None, pred=[], probability=[], test_log=None, id_list=None):
        self.data = None
        self.label = None
        self.transform = transform
        self.strong_aug = transform
        self.mode = mode
        self.pred = pred
        self.probability = None
        self.real_label = real_label
        self.id_list = id_list
        self.args = args

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

        self.data_num = len(self.data)

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.data[index], self.label[index], self.probability[index]
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


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, args, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.args = args

        self.train_data, self.train_label, self.noise_label, self.test_data, self.test_label \
            = initial_data(self.dataset, self.r, self.noise_mode, self.root_dir, args)

        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def mapping_func(self, w):
        beta = self.args.beta
        def beta_f(t):
            return math.pow(1 - t, beta - 1)

        v, _ = integrate.quad(beta_f, 0, 1)
        tilde_w = 1 / v * math.pow(1 - w, beta - 1)
        return tilde_w

    def run(self, mode, pred=[], prob=[], test_log=None):
        if mode == 'warmup':
            all_dataset = cifar_dataset(self.args, self.train_data, self.train_label, self.noise_label, self.transform_train, mode='all', strong_transform=None,
                                            pred=pred, probability=prob, test_log=test_log)
            trainloader = DataLoader(dataset=all_dataset, batch_size=128, shuffle=True, num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(self.args, self.train_data, self.train_label, self.noise_label, self.transform_train, mode='labeled',
                                            strong_transform=None, pred=pred, probability=prob, test_log=test_log)

            data_num = labeled_dataset.data_num  # a list with size of arg.num_class
            labels = labeled_dataset.label

            # calculate per class number
            cls_number = [0] * self.args.num_class
            for i in range(data_num):
                cls_number[labels[i]] += 1

            # class number normalization, get v
            cls_number = np.array(cls_number, dtype='float32').reshape(1, -1)
            v = normalize(cls_number, norm='l1').tolist()[0]   # v vector in the paper

            # get tilde_v by mapping function
            tilde_v = [0] * self.args.num_class
            for i in range(self.args.num_class):
                tilde_v[i] = self.mapping_func(v[i])

            # customize the sampler
            weight_ = [0] * data_num    # assign weight to each training sample
            weight_reversed = [0] * data_num
            for i in range(data_num):
                weight_[i] = v[labels[i]]
                weight_reversed[i] = tilde_v[labels[i]]

            sampler_v = WeightedRandomSampler(weight_, num_samples=data_num, replacement=False)
            sampler_v_tilde = WeightedRandomSampler(weight_reversed, num_samples=data_num, replacement=True)

            trainloader_v = DataLoader(labeled_dataset, self.batch_size, sampler=sampler_v, shuffle=False, num_workers=4, drop_last=True)
            trainloader_v_tilde = DataLoader(labeled_dataset, self.batch_size, sampler=sampler_v_tilde, shuffle=False, num_workers=4, drop_last=True)
            return trainloader_v, trainloader_v_tilde

        elif mode == 'test':
            test_dataset = cifar_dataset(self.args, self.test_data, self.train_label, self.test_label, self.transform_train, mode='test',
                                         strong_transform=None, pred=pred, probability=prob)
            test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(self.args, self.train_data, self.train_label, self.noise_label, self.transform_train, mode='all',
                                         strong_transform=None, pred=pred, probability=prob)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            return eval_loader