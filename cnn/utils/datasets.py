import torch
import torchvision
import torchvision.transforms as transforms

# --- Dataset Specifications ---
# Mean and Std Dev for CIFAR-10 as required
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def get_cifar10_dataloaders(batch_size=128, data_root='./cnn/data'):
    """
    Prepares and returns the CIFAR-10 training and testing DataLoaders.
    Applies required normalization and data augmentation.
    """
    print(f"Loading CIFAR-10 data to/from: {data_root}")

    # 3. Dataset Specifications - Training Augmentation
    transform_train = transforms.Compose([
        # Random Crop (32x32, with padding=4)
        transforms.RandomCrop(32, padding=4),
        # Random Horizontal Flip
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalization
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 3. Dataset Specifications - Test Set (only normalization)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # Normalization
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader

if __name__ == '__main__':
    # Example usage (will download data if not present)
    train_dl, test_dl = get_cifar10_dataloaders(batch_size=128)
    print(f"\nTrain Batches: {len(train_dl)}")
    print(f"Test Batches: {len(test_dl)}")
    data, labels = next(iter(train_dl))
    print(f"Sample Batch Shape: {data.shape}")