import os
import torch
from torchvision import datasets, transforms

def getData(dataset):

    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

        input_dim = 3*32*32
        output_dim = 10

    elif dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)

        input_dim = 3*32*32
        output_dim = 100

    elif dataset == "mnist":
        mean = (0.1307,)
        std = (0.3081,)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST('./data', train=False, download=True, transform=transform)

        input_dim = 1*28*28
        output_dim = 10

    elif dataset == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        traindir = os.path.join(os.getenv('IMAGENET_PATH'), "train")
        valdir = os.path.join(os.getenv('IMAGENET_PATH'), "val")

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))

        input_dim = 3*224*224
        output_dim = 1000
    
    return trainset, testset, input_dim, output_dim
