from torch.utils.data import DataLoader
from torchvision import datasets, transforms


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Correct normalisation values for CIFAR-10
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Correct normalisation values for CIFAR-10
])

# Download and load the training set
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Download and load the test set
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)