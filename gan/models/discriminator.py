# gan/models/discriminator.py

import torch.nn as nn

# Model parameters
FEATURES_D = 64 # Number of feature maps in the discriminator

class Discriminator(nn.Module):
    """
    DCGAN Discriminator that classifies a 32x32x3 image as real or fake.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Input is (N, 3, 32, 32)
            nn.Conv2d(3, FEATURES_D, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (N, FEATURES_D, 16, 16)
            
            nn.Conv2d(FEATURES_D, FEATURES_D * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_D * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (N, FEATURES_D*2, 8, 8)
            
            nn.Conv2d(FEATURES_D * 2, FEATURES_D * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_D * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (N, FEATURES_D*4, 4, 4)
            
            nn.Conv2d(FEATURES_D * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Final state size: (N, 1, 1, 1) -> output is a single probability score
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)