import torch
import torch.nn as nn
import os
from cnn.models.resnet import ResNet18 # Assuming ResNet18 class is available

def load_cnn_feature_extractor(checkpoint_path='./cnn/checkpoints/resnet_best.pth'):
    """
    Loads the trained ResNet-18 model and prepares it for feature extraction.
    The final classification layer is removed.
    """
    # 1. Instantiate the full model (Use the ResNet18 class, not the factory function)
    # The architecture must match the checkpoint weights.
    model = ResNet18() 
    
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Please train the CNN first.")

    # 2. Load the trained weights
    # Map location ensures weights load correctly regardless of training device (CPU/CUDA)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # 3. Modify the model: Remove the final classification layer (linear layer)
    # The output will now be the features right before the classification head.
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    # 4. Set to evaluation mode
    feature_extractor.eval()
    
    # Freeze weights (we don't want to train the feature extractor)
    for param in feature_extractor.parameters():
        param.requires_grad = False
        
    print(f"Loaded trained ResNet-18 features from {checkpoint_path}")

    return feature_extractor

if __name__ == '__main__':
    # This example requires the checkpoint to exist
    try:
        extractor = load_cnn_feature_extractor(
            checkpoint_path='../checkpoints/resnet_best.pth' # Adjusted path for internal test
        )
        print(f"Feature Extractor Layers: {extractor}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        features = extractor(dummy_input)
        print(f"Feature Output Shape (should be 512, 1, 1): {features.squeeze().shape}")
    except FileNotFoundError as e:
        print(f"Test failed: {e}")