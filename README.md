👋 Welcome to FusionNet: Our Dual Deep Learning Project!
Hey there! Thanks for checking out FusionNet. This project is our ambitious playground where we're tackling two of the coolest problems in AI at once: teaching a machine to see (Classification) and teaching a machine to dream (Generation).

We’re building two separate, high-powered modules that both work with the well-known CIFAR-10 image dataset.

🌟 What We're Building
The main goal here is to create a super clean and organized code structure that makes collaboration easy. We want a pro-level framework where we can grow both models without them stepping on each other's toes.

🧭 How We Organized the Code
We kept things simple and separated into two main folders:

FusionNet/
├── cnn/                             # 🧠 The Brain: Image Classification Module (ResNet-18)
│   ├── models/                      
│   ├── train_cnn.py                 # Your Mission! Training the classifier.
│   └── config.json                  # All the numbers and settings for the CNN model.
│
├── gan/                             # 🎨 The Artist: Image Generation Module (DCGAN)
│   ├── config.json                  # All the numbers and settings for the GAN model.
│   ├── models/                      
│   ├── train_gan.py                 # The fully coded training script.
│   └── *Results* # Where the plots and generated images land!
│
└── README.md


🧠 The CNN Classifier (ResNet-18)
This is the part that needs your magic!

The Mission: Teach our model to correctly label those small CIFAR-10 images (is it a dog? a truck? a bird?).

The Engine: We're using ResNet-18, which is a powerhouse for classification. It uses special "skip connections" that help the model learn deeply without forgetting what it learned earlier.

Proof It Works: We need a classic Train vs. Test Accuracy graph to show how well it's performing as it learns.

🎨 The GAN Generator (DCGAN) - (Done & Ready!)
This part of the code is complete and has been tested on a CPU.

The Mission: To create entirely new, realistic-looking CIFAR-10 images from scratch!

The Architecture: We used a DCGAN, which is like a competitive game between two networks: one network generates images, and the other judges them.

The Fixes: We've implemented Label Smoothing and a JSON Config to make the training as stable and professional as possible.

Proof It Works (The Plots are Ready):

Image Output: We generate a grid of 64 brand-new images.

The Required Graph: We track the Discriminator's Accuracy over time. This shows how hard the Generator is working to fool the system—when the accuracy dips toward 50%, we know we've achieved near-perfect artificial realism!

🚀 How to Get Started
1. The Right Environment (Crucial!)
Use a GPU: This project needs the speed! If you have an RTX 3050 or better, please make sure PyTorch is installed with CUDA support. The code will automatically use the GPU if it detects it.

Dependencies: Standard Python/PyTorch setup (see the requirements.txt file we'll generate later).

2. Training the GAN (My Side)
This module uses a separate config.json to keep all the settings outside the code.

Check gan/config.json: Feel free to look, but try not to change the GAN settings unless we discuss it!

Run the script: Always run from the root directory to ensure the imports work:

Bash

# Must be run from the FusionNet/ (project root) directory
python -m gan.train_gan
3. Training the CNN (Your Side)
You'll need to create the training loop in cnn/train_cnn.py and potentially set up a cnn/config.json. Good luck!
