import os
from torchvision import datasets, transforms

def getData():

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
