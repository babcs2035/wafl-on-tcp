import os
import warnings

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# Constants for directories and hyperparameters
DATA_DIR = "data"
MODELS_DIR = "models"
OUTS_DIR = "outs"
BATCH_SIZE = 500
N_OUTPUT = 10
N_HIDDEN = 128
LR = 0.01
N_EPOCHS = 16


class Net(nn.Module):
    """
    Neural network model with one hidden layer.
    """

    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        # Define layers
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3


def setup():
    """
    Setup function to initialize device, directories, and data loaders.
    """
    global device, train_loader, test_loader

    # Configure matplotlib and numpy
    plt.rcParams["font.size"] = 14
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["axes.grid"] = True
    np.set_printoptions(suppress=True, precision=5)

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.simplefilter("ignore")
    print(f"🔧 Using device: {device}")

    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTS_DIR, exist_ok=True)
    print(f"📂 Directories created: {DATA_DIR}, {MODELS_DIR}, {OUTS_DIR}")

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image
        ]
    )

    # Load MNIST dataset
    train_set = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    print(
        f"📊 Loaded datasets: {len(train_set)} training samples, {len(test_set)} testing samples"
    )

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"🚚 Data loaders created with batch size: {BATCH_SIZE}")


def train():
    """
    Train the neural network model.
    """
    global device, train_loader, test_loader, history

    # Initialize the model
    for images, labels in train_loader:
        break
    net = Net(images[0].numpy().shape[0], N_OUTPUT, N_HIDDEN).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    print(f"🧠 Model initialized with {N_HIDDEN} hidden units and learning rate: {LR}")
    print(summary(net))

    # Initialize history for tracking metrics
    history = np.zeros((0, 5))

    # Training loop
    for epoch in range(N_EPOCHS):
        print(f"🚀 Starting epoch {epoch + 1}/{N_EPOCHS}")
        n_train_acc, n_val_acc = 0, 0
        train_loss, val_loss = 0, 0
        n_train, n_test = 0, 0

        # Training phase
        for inputs, labels in tqdm(train_loader, desc="Training"):
            train_batch_size = len(labels)
            n_train += train_batch_size
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = torch.max(outputs, 1)[1]
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()

        # Validation phase
        for inputs_test, labels_test in test_loader:
            test_batch_size = len(labels_test)
            n_test += test_batch_size
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # Forward pass
            outputs_test = net(inputs_test)
            loss_test = criterion(outputs_test, labels_test)
            predicted_test = torch.max(outputs_test, 1)[1]

            # Calculate accuracy
            val_loss += loss_test.item() * test_batch_size
            n_val_acc += (predicted_test == labels_test).sum().item()

        # Calculate metrics
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        ave_train_loss = train_loss / n_train
        ave_val_loss = val_loss / n_test
        print(
            f"📈 Epoch [{epoch+1}/{N_EPOCHS}], "
            f"Train Loss: {ave_train_loss:.5f}, Train Acc: {train_acc:.5f}, "
            f"Val Loss: {ave_val_loss:.5f}, Val Acc: {val_acc:.5f}"
        )

        # Save metrics to history
        item = np.array([epoch + 1, ave_train_loss, train_acc, ave_val_loss, val_acc])
        history = np.vstack((history, item))

        # Save model checkpoint
        torch.save(
            net.state_dict(),
            os.path.join(MODELS_DIR, f"model_epoch_{epoch+1}.pth"),
        )
        print(f"💾 Model checkpoint saved for epoch {epoch + 1}")

    print("✅ Training completed and all models saved.")


def result():
    """
    Generate and save training results as plots.
    """
    global history

    # Plot loss
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.plot(history[:, 0], history[:, 1], "b", label="train")
    plt.plot(history[:, 0], history[:, 3], "k", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.savefig(os.path.join(OUTS_DIR, "loss.png"))
    print("📊 Loss plot saved as 'loss.png'")
    plt.clf()

    # Plot accuracy
    plt.rcParams["figure.figsize"] = (9, 8)
    plt.plot(history[:, 0], history[:, 2], "b", label="train")
    plt.plot(history[:, 0], history[:, 4], "k", label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.savefig(os.path.join(OUTS_DIR, "accuracy.png"))
    print("📊 Accuracy plot saved as 'accuracy.png'")


if __name__ == "__main__":
    print("🔧 Setting up the environment...")
    setup()
    print("🚀 Starting training process...")
    train()
    print("📊 Generating results...")
    result()
    print("🎉 All tasks completed successfully!")
