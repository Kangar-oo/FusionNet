# gan/train_gan.py

import sys
import os

sys.path.append('.') 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Import custom files (These lines stay the same)
from gan.models.generator import Generator, LATENT_DIM, weights_init
from gan.models.discriminator import Discriminator
from gan.utils.datasets import get_cifar10_gan_dataloader

# --- Configuration ---
BATCH_SIZE = 128
NUM_EPOCHS = 25
LR = 0.0002
BETA1 = 0.5 # Hyperparameter for Adam optimizers

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Setup Models and Data ---
def setup_gan():
    # 1. Data
    dataloader = get_cifar10_gan_dataloader(BATCH_SIZE)
    
    # 2. Models
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    # Apply custom weights initialization
    netG.apply(weights_init)
    netD.apply(weights_init)

    # 3. Loss and Optimizers
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    # Noise vector used to track G's progress during training
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    
    return dataloader, netG, netD, criterion, optimizerD, optimizerG, fixed_noise

# --- 2. Training Loop ---
def run_gan_experiment():
    dataloader, netG, netD, criterion, optimizerD, optimizerG, fixed_noise = setup_gan()

    # Establish conventions for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    print(f"Starting GAN training on {device} for {NUM_EPOCHS} epochs.")
    
    G_losses = []
    D_losses = []
    img_list = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)
            
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, LATENT_DIM, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach()) # Detach G from D training
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Combine losses and step
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # G wants to fool D, so label its output as 'real'
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()
            
            # Save Losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Checkpoint images and progress at the end of each epoch
        if epoch % 5 == 0 or epoch == NUM_EPOCHS:
            with torch.no_grad():
                fake_images = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_images, padding=2, normalize=True))

            print(f'[{epoch}/{NUM_EPOCHS}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z2:.4f}')

    # --- 3. Save Final Models ---
    torch.save(netG.state_dict(), './gan/checkpoints/G_weights.pth')
    torch.save(netD.state_dict(), './gan/checkpoints/D_weights.pth')
    print("GAN training complete. Models saved to checkpoints.")

    # --- 4. Plot Loss History (Accuracy Graph Plot analog) ---
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./gan/gan_loss_plot.png')
    plt.show()

    # --- 5. Generate Fake Image (Final Output) ---
    final_fake_grid = vutils.make_grid(img_list[-1], padding=2, normalize=True)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Final Generated Images (Fake Image)")
    plt.imshow(np.transpose(final_fake_grid,(1,2,0)))
    plt.savefig('./gan/final_generated_images.png')
    plt.show()

if __name__ == '__main__':
    # Make sure all directories exist
    import os
    os.makedirs('gan/checkpoints', exist_ok=True)
    os.makedirs('gan/data', exist_ok=True)
        
    run_gan_experiment()