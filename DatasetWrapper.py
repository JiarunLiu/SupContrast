import torch
import numpy as np
import torch.nn as nn
import torchvision

from noisy_dataset import noisify




class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset, noise_type='clean', noise_rate=0):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate

        # noisify labels
        train_clean_labels = np.expand_dims(np.asarray(dataset.targets), 1)
        train_noisy_labels, _ = noisify(train_labels=train_clean_labels,
                                        nb_classes=10,
                                        noise_type=noise_type,
                                        noise_rate=noise_rate)
        self.train_noisy_labels = train_noisy_labels.flatten().tolist()
        assert len(self.train_noisy_labels) == len(dataset.targets)

        self.noise_or_not = (np.transpose(
            self.train_noisy_labels) == np.transpose(train_clean_labels)).squeeze()


    def save_noise_labels(self, dir):
        np.save(dir, np.asarray(self.train_noisy_labels))

    def __getitem__(self, index):
        img, target = self.dataset[index]
        not_noise = self.noise_or_not[index]
        return img, target, not_noise

    def __len__(self):
        return len(self.dataset)

#
# if __name__ == '__main__':
#     from torchvision import transforms, datasets
#     from util import TwoCropTransform
#
#     train_dataset = datasets.CIFAR10(root="./data/cifar10",
#                                      transform=TwoCropTransform(
#                                          transforms.ToTensor()),
#                                      download=True)
#
#     train_dataset = DatasetWrapper(train_dataset, "symmetric", 0.1)
#     train_dataset.save_noise_labels('y.npy')
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=16, num_workers=0)
#
#     for i, (img, target, not_noise) in enumerate(train_loader):
#         print("Img: {}\tTarget: {}\tIS_NOISE: {}".format(img.shape, target.shape, is_noise.shape))