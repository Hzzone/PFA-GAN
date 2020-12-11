import numpy as np
import torch.utils.data as tordata
import os.path as osp
from ops import age2group
from torchvision.datasets.folder import pil_loader
import random
import torch


class data_prefetcher():
    def __init__(self, loader, *norm_index):
        self.loader = iter(loader)
        self.normlize = lambda x: x.sub_(0.5).div_(0.5)
        self.norm_index = norm_index
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = [
                self.normlize(x.cuda(non_blocking=True)) if i in self.norm_index else x.cuda(non_blocking=True)
                for i, x in enumerate(self.next_input)]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


def load_source(dataset_name, train=True, age_group=4):
    data_root = osp.join(osp.dirname(osp.abspath(__file__)), 'materials', dataset_name)
    with open(osp.join(data_root, '{}.txt'.format('train' if train else 'test')), 'r') as f:
        source = np.array([x.strip().split() for x in f.readlines()])
    path = np.array([osp.join(data_root, x[0]) for x in source])
    age = np.array([int(x[1]) for x in source])
    group = age2group(age, age_group)
    return {'path': path, 'age': age, 'group': group}


class BaseDataset(tordata.Dataset):
    def __init__(self,
                 dataset_name,
                 age_group,
                 train=False,
                 max_iter=0,
                 batch_size=0,
                 transforms=None):
        self.dataset_name = dataset_name
        self.age_group = age_group
        self.train = train
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.total_pairs = batch_size * max_iter
        self.transforms = transforms

        data = load_source(train=train, dataset_name=dataset_name, age_group=age_group)
        self.image_list, self.ages, self.groups = data['path'], data['age'], data['group']

        self.mean_ages = np.array([np.mean(self.ages[self.groups == i])
                                   for i in range(self.age_group)]).astype(np.float32)
        self.label_group_images = []
        self.label_group_ages = []
        for i in range(self.age_group):
            self.label_group_images.append(
                self.image_list[self.groups == i].tolist())
            self.label_group_ages.append(
                self.ages[self.groups == i].astype(np.float32).tolist())

    def __len__(self):
        return self.total_pairs


class PFADataset(BaseDataset):
    def __init__(self,
                 dataset_name,
                 age_group,
                 max_iter,
                 batch_size,
                 source,
                 transforms=None,
                 **kwargs):
        super(PFADataset, self).__init__(
            dataset_name=dataset_name,
            age_group=age_group,
            batch_size=batch_size,
            max_iter=max_iter,
            transforms=transforms)
        # define label_pairs
        np.random.seed(0)
        if dataset_name == 'cacd':
            import itertools
            self.target_labels = np.random.randint(source + 1, self.age_group, self.total_pairs)
            pairs = np.array(list(itertools.combinations(range(age_group), 2)))
            p = [1, 1, 1, 0.5, 0.5, 0.5]
            p = np.array(p) / np.sum(p)
            pairs = pairs[
                    np.random.choice(range(len(pairs)), self.total_pairs, p=p), :]
            source_labels, target_labels = pairs[:, 0], pairs[:, 1]
            self.source_labels = source_labels
            self.target_labels = target_labels
        elif dataset_name == 'morph':
            self.source_labels = np.ones(self.total_pairs, dtype=int) * source
            self.target_labels = np.random.randint(source + 1, self.age_group, self.total_pairs)

        self.true_labels = np.random.randint(0, self.age_group, self.total_pairs)

    def __getitem__(self, idx):
        source_label = self.source_labels[idx]
        target_label = self.target_labels[idx]
        true_label = self.true_labels[idx]

        source_img = pil_loader(random.choice(self.label_group_images[source_label]))

        index = random.randint(0, len(self.label_group_images[true_label]) - 1)
        true_img = pil_loader(self.label_group_images[true_label][index])
        true_age = self.label_group_ages[true_label][index]
        mean_age = self.mean_ages[target_label]

        if self.transforms is not None:
            source_img = self.transforms(source_img)
            true_img = self.transforms(true_img)
        return source_img, true_img, source_label, target_label, true_label, true_age, mean_age


class GroupDataset(BaseDataset):
    def __init__(self,
                 dataset_name,
                 age_group,
                 train,
                 group,
                 transforms=None):
        super(GroupDataset, self).__init__(
            dataset_name=dataset_name,
            train=train,
            age_group=age_group,
            transforms=transforms)
        self.label_group_image = self.label_group_images[group]

    def __getitem__(self, idx):
        img = pil_loader(self.label_group_image[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.label_group_image)


class AgeDataset(BaseDataset):
    def __init__(self,
                 dataset_name,
                 age_group,
                 train,
                 transforms=None):
        super(AgeDataset, self).__init__(
            dataset_name=dataset_name,
            train=train,
            age_group=age_group,
            transforms=transforms)

    def __getitem__(self, idx):
        img = pil_loader(self.image_list[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.ages[idx]

    def __len__(self):
        return len(self.image_list)
