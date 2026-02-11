
# 🚀 Welcome to FusionNet: Our Dual Deep Learning Lab!
Hello! This repository hosts FusionNet, our exciting project where we are building two complementary, high-performance deep learning systems that work on the CIFAR-10 image dataset. Think of it as a collaboration between a brilliant scientist (the CNN) and a talented artist (the GAN).

The main goal is to create a super clean, modular framework that allows both parts to be developed and upgraded independently.

# 🧠 Module 1: The CNN Classifier (ResNet-18)

Your Mission: Achieve high accuracy in classifying the 10 classes of CIFAR-10 (dogs, cats, trucks, etc.).

The Architecture: We're leveraging the robust ResNet-18 architecture. Its "residual blocks" are key to training very deep networks stably.

The Proof: The main deliverable here will be the classic Train vs. Test Accuracy Plot, showing how well our model generalizes to new data.

# 🎨 Module 2: The GAN Generator (DCGAN)
(This module is 100% coded, tested, and ready for integration.)

The Mission: Create a model that can "dream up" novel images that look like they belong in the CIFAR-10 dataset.

The Setup: We used a Deep Convolutional GAN (DCGAN) , implementing essential tricks like Label Smoothing to keep the training game balanced and prevent the model from crashing.

The Key to Modularity: All the hyperparameters (NUM_EPOCHS, BATCH_SIZE, etc.) are stored cleanly in the gan/config.json file.

The Proof (Evaluation Complete!):

Generated Images: A grid of 64 new images showing the model's creative output.

Required Graph: We track the Discriminator's Accuracy. This plot shows that the Generator is continually improving and forcing the Discriminator's accuracy down towards the ideal 50% equilibrium .

# 🚀 Getting Started & Running the Code
1. Prerequisites (The Essentials)
Python 3.8+

PyTorch (CUDA recommended): If you're using a GPU (like the RTX 3050), make sure you have the CUDA-enabled version of PyTorch installed for massive speed-ups!

A requirements.txt file (to be added) will list all exact packages.

2. Running My Module (For Testing)
To run the GAN module and see the results, you must execute the script from the root directory of the project (FusionNet/).
# Run this from the main project folder (CNN-ResNet/)
python -m gan.train_gan
3. Running Your Module (The Next Step!)
Once you've implemented the ResNet logic, you can run your part similarly:
# Run this from the main project folder (CNN-ResNet/)
python -m cnn.train_cnn


Check Config: Take a look at gan/config.json to see the current training parameters (currently set to 50 epochs and a batch size of 128 for GPU efficiency).

Execute:
# Run this from the main project folder (CNN-ResNet/)
python -m gan.train_gan
3. Running Your Module (The Next Step!)
Once you've implemented the ResNet logic, you can run your part similarly:
# Run this from the main project folder (CNN-ResNet/)
python -m cnn.train_cnn
