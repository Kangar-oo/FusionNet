import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# 6. Imports: Must be relative to the project root
from CNN.models.resnet import ResNet18_cifar10
from CNN.utils.datasets import get_cifar10_dataloaders

# 5. Training and Evaluation Specifications
NUM_EPOCHS = 10 
CHECKPOINT_DIR = './checkpoints'  # This creates it in your current folder
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'resnet_best.pth')
PLOT_PATH = './accuracy_graph_plot_resnet.png'

# --- Utility Functions ---

def adjust_learning_rate(optimizer, epoch, initial_lr=0.1):
    """
    Simple decay: reduce LR by 10x at epochs 20 and 40.
    """
    lr = initial_lr
    if epoch >= 40:
        lr /= 100
    elif epoch >= 20:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(model, path):
    """Saves the best model state dictionary."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"-> Saved best model checkpoint to {path}")

def plot_accuracy(train_acc_history, test_acc_history, path):
    """
    Generates and saves a visual graph plot of accuracy over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_history, label='Train Accuracy', marker='o')
    plt.plot(test_acc_history, label='Test Accuracy', marker='o')
    plt.title('Train vs. Test Accuracy (ResNet-18)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    print(f"-> Saved accuracy plot to {path}")

# --- Core Training and Testing Loops ---

def train(model, dataloader, optimizer, criterion, device):
    """Performs one epoch of training."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, dataloader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# --- Main Execution ---

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Data
    trainloader, testloader = get_cifar10_dataloaders(batch_size=128)

    # 2. Model, Loss, and Optimizer
    model = ResNet18_cifar10().to(device)
    
    # Loss Function: CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: SGD with Momentum (0.1, 0.9, 5e-4)
    initial_lr = 0.1
    optimizer = optim.SGD(
        model.parameters(), 
        lr=initial_lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )

    # 3. Training Loop and History
    train_acc_history = []
    test_acc_history = []
    best_test_accuracy = 0.0

    print("\nStarting Training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Apply Learning Rate Schedule
        current_lr = adjust_learning_rate(optimizer, epoch, initial_lr)

        # Train
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, device)
        
        # Test
        test_loss, test_acc = test(model, testloader, criterion, device)

        # Save History
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Save Best Checkpoint
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            save_checkpoint(model, CHECKPOINT_PATH)
            
    print("\nTraining Finished.")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")

    # 4. Save Final Plot (Crucial Requirement)
    plot_accuracy(train_acc_history, test_acc_history, PLOT_PATH)

if __name__ == '__main__':
    main()