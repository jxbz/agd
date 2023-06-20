from torchvision import datasets, transforms

def getData():

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
    
    return trainset, testset, input_dim, output_dim
