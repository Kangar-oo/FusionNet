# gan/utils/datasets.py

import torch
from torchvision import datasets, transforms

# Size of the CIFAR-10 images
IMAGE_SIZE = 32

def get_cifar10_gan_dataloader(batch_size=128):
    """
    Loads and preprocesses the CIFAR-10 dataset for GAN training.
    Normalizes images to the range [-1, 1].
    """
    
    transform = transforms.Compose([
        # Resize is not strictly necessary for CIFAR-10 (already 32x32), but good practice
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        # Normalize to [-1, 1] range: (x - 0.5) * 2
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])

    # Load Dataset, placing it in the gan/data folder
    dataset = datasets.CIFAR10(
        root='./gan/data', train=True, download=True, transform=transform
    )
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return dataloader

# Example usage (uncomment to test):
# if __name__ == '__main__':
#     dataloader = get_cifar10_gan_dataloader()
#     print(f"Number of training images: {len(dataloader.dataset)}")