import os
import time
import torch
import argparse
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from noisy_dataset import noisify

from networks.resnet_big import SupConResNet, LinearClassifier, SupCEResNet

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    if opt.noise > 0:
        # noisify labels
        train_labels = np.expand_dims(np.asarray(train_dataset.targets), 1)
        train_noisy_labels, _ = noisify(train_labels=train_labels,
                                        nb_classes=10,
                                        noise_type="symmetric",
                                        noise_rate=opt.noise)
        train_noisy_labels = train_noisy_labels.flatten().tolist()
        assert len(train_noisy_labels) == len(train_dataset.targets)
        train_dataset.targets = train_noisy_labels

    if opt.yfile is not None:
        train_noisy_labels = np.load(opt.yfile)
        train_noisy_labels = train_noisy_labels.tolist()
        assert len(train_noisy_labels) == len(train_dataset.targets)
        train_dataset.targets = train_noisy_labels

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_model(opt):
    if "SupCE" in opt.ckpt:
        model = SupConResNet(name=opt.model, num_classes=opt.n_cls)
    else:
        model = SupConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(opt.device)
        classifier = classifier.to(opt.device)
        criterion = criterion.to(opt.device)
        # cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion

@torch.no_grad()
def gen_single_features(args):
    model, classifier, criterion = set_model(args)
    train_loader, val_loader = set_loader(args)

    model.eval()

    # guess feature num
    for i, (input, target) in enumerate(train_loader):
        input = input.to(args.device)
        feature = model(input)
        break
    num_features = feature.shape[1]

    features = np.zeros((len(train_loader.dataset), num_features), dtype=np.float32)
    labels = np.zeros(len(train_loader.dataset), dtype=np.long)
    batch_num = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        index = np.arange(i*args.batch_size, i*args.batch_size + len(target))
        input = input.to(args.device)
        if "SupCE" in args.ckpt:
            feature = model.encoder(input)
        else:
            feature = model(input)
        features[index] = feature.cpu().numpy()
        labels[index] = target
        print("\rget clustering data: [{}/{}]".format(i+1, batch_num), end='')
    print("\nFinish collect cluster data.")

    torch.cuda.empty_cache()

    if args.label != 0:
        print("Filtering labels")
        print(features.shape)
        ind = np.where(labels == 0)[0]
        features = features[ind]
        labels = labels[ind]
        print(features.shape)

    if args.reduce:
        print("reduce sample number")
        features = features[:args.reduce]
        labels = labels[:args.reduce]

    return features, labels

def tsne(X, dim=2):
    tsne = manifold.TSNE(n_components=dim, init='pca')
    print("Fitting...")
    X_tsne = tsne.fit_transform(X)
    print("Finished.")
    return X_tsne

def plot_tsne(args):
    X, y = gen_single_features(args)
    X_tsne = tsne(X, dim=2)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    print("Start plot...")
    # cm = ['orange', 'blue']
    # cm = ['tomato', 'cadetblue']
    for i in range(X_norm.shape[0]):
        # plt.scatter(X_norm[i,0], X_norm[i,1], c=plt.cm.Set1(y[i]))
        # plt.scatter(X_norm[i,0], X_norm[i,1], c=cm[y[i]], s=10)
        plt.scatter(X_norm[i, 0], X_norm[i, 1], c=plt.cm.Set1(y[i]), s=6, alpha=0.4)


    plt.xticks([])
    plt.yticks([])
    plt.title(args.title)
    if args.save_fig:
        plt.savefig(os.path.join(args.save_fig, "{}.png".format(args.title)), dpi=800)
    else:
        plt.show()
    print("Done")


def get_args():
    parser = argparse.ArgumentParser(description='Visual recording data')
    parser.add_argument('--dataset', default="cifar10", type=str, help='dataset')
    parser.add_argument('--device', default="cuda:0", type=str, help='device')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--yfile', default=None, type=str)
    parser.add_argument('--reduce', default=2000, type=int)
    parser.add_argument('--batch_size', '-b', default=32, type=int)
    parser.add_argument('--label', default=0, type=int)
    parser.add_argument('--noise', dest='noise', default=0, type=float)
    parser.add_argument('--title', default='epoch 0', type=str)
    parser.add_argument('--ckpt', type=str, help='path to pre-trained model',
    default='./save/SupCon/cifar10_models/SupCon_cifar10_0.0_resnet50_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_0/last.pth')
    parser.add_argument('--save-fig', default=None, type=str)
    # parser.add_argument('--save-fig', default="./table/tsne", type=str)
    args = parser.parse_args()

    args.data_folder = "./data/cifar10"
    args.n_cls = 10

    return args


if __name__ == "__main__":
    start = time.time()
    args = get_args()
    plot_tsne(args)
    end = time.time()
    use = (end-start)/60
    print("using {} min".format(use))
