from torchvision import datasets, transforms

def getData():
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

    return trainset, testset, input_dim, output_dim
