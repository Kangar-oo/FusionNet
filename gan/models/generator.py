# gan/models/generator.py

import torch.nn as nn

# Model parameters
LATENT_DIM = 100 # Size of the input noise vector (z)
FEATURES_G = 64  # Number of feature maps in the generator

class Generator(nn.Module):
    """
    DCGAN Generator that maps a latent vector (noise) to a 32x32x3 image.
    Uses Transposed Convolution for upsampling.
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Start with a projection from latent space: (N, LATENT_DIM, 1, 1)
            nn.ConvTranspose2d(LATENT_DIM, FEATURES_G * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES_G * 4),
            nn.ReLU(True),
            # State size: (N, FEATURES_G*4, 4, 4)
            
            nn.ConvTranspose2d(FEATURES_G * 4, FEATURES_G * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_G * 2),
            nn.ReLU(True),
            # State size: (N, FEATURES_G*2, 8, 8)
            
            nn.ConvTranspose2d(FEATURES_G * 2, FEATURES_G, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_G),
            nn.ReLU(True),
            # State size: (N, FEATURES_G, 16, 16)
            
            nn.ConvTranspose2d(FEATURES_G, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final state size: (N, 3, 32, 32) (RGB image)
        )

    def forward(self, input):
        # Input noise vector (z) is of shape (N, LATENT_DIM)
        # We need to reshape it to (N, LATENT_DIM, 1, 1) for ConvTranspose2d
        return self.main(input.view(-1, LATENT_DIM, 1, 1))

# Helper to initialize weights for the GAN (DCGAN paper recommendation)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)