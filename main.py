import os
import pickle
import socket
import warnings

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Constants for directories and hyperparameters
CONFIG = {
    "data_dir": "data",
    "models_dir": "models",
    "outs_dir": "outs",
    "batch_size": 500,
    "n_output": 10,
    "n_hidden": 128,
    "lr": 0.01,
    "n_epochs": 16,
    "buffer_size": 4096,
    "host": "127.0.0.1",
    "port": 5000,
    "independent_epochs": 5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")


class Net(nn.Module):
    """
    Neural network model with one hidden layer.
    """

    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x


def setup_data():
    """
    Prepare the MNIST dataset and return DataLoader objects for training and testing.
    """
    print("📦 Setting up data loaders...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_set = datasets.MNIST(
        root=CONFIG["data_dir"], train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=CONFIG["data_dir"], train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False)
    print("✅ Data loaders are ready.")
    return train_loader, test_loader


def save_model(net, save_dir, epoch, role=""):
    """
    Save the model checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = (
        f"{role}_model_epoch_{epoch+1}.pth" if role else f"model_epoch_{epoch+1}.pth"
    )
    torch.save(net.state_dict(), os.path.join(save_dir, filename))
    print(f"💾 Model saved: {filename}")


def send_data(sock, data):
    """
    Send serialized data via TCP.
    """
    serialized_data = pickle.dumps(data)
    sock.sendall(len(serialized_data).to_bytes(8, "big"))
    sock.sendall(serialized_data)
    print("📤 Data sent.")


def receive_data(sock):
    """
    Receive serialized data via TCP.
    """
    data_size = int.from_bytes(sock.recv(8), "big")
    data = b""
    while len(data) < data_size:
        packet = sock.recv(CONFIG["buffer_size"])
        if not packet:
            break
        data += packet
    print("📥 Data received.")
    return pickle.loads(data)


def train_model(
    net,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    n_epochs,
    save_dir,
    is_multi=False,
    conn=None,
    role="",
):
    """
    Train the model and optionally perform collaborative learning.
    """
    history = []

    for epoch in range(n_epochs):
        print(f"🔄 Epoch {epoch + 1}/{n_epochs}")
        net.train()
        train_loss, train_correct = 0, 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

        if is_multi and epoch >= CONFIG["independent_epochs"] and conn:
            if role == "server":
                send_data(conn, net.l1.state_dict())
                net.l1.load_state_dict(receive_data(conn))
            else:
                net.l1.load_state_dict(receive_data(conn))
                send_data(conn, net.l1.state_dict())

        net.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss /= len(test_loader.dataset)
        val_acc = val_correct / len(test_loader.dataset)
        history.append((epoch + 1, train_loss, train_acc, val_loss, val_acc))

        save_model(net, save_dir, epoch, role)
        print(
            f"📊 Epoch {epoch+1}/{n_epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

    return np.array(history)


def plot_results(single_history, multi_history, save_dir):
    """
    Plot and save the comparison of training results between single and multi modes.
    """
    print("📈 Plotting results...")
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(single_history[:, 0], single_history[:, 1], label="Single Train")
    plt.plot(single_history[:, 0], single_history[:, 3], label="Single Val")
    plt.plot(multi_history[:, 0], multi_history[:, 1], label="Multi Train")
    plt.plot(multi_history[:, 0], multi_history[:, 3], label="Multi Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Comparison")

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(single_history[:, 0], single_history[:, 2], label="Single Train")
    plt.plot(single_history[:, 0], single_history[:, 4], label="Single Val")
    plt.plot(multi_history[:, 0], multi_history[:, 2], label="Multi Train")
    plt.plot(multi_history[:, 0], multi_history[:, 4], label="Multi Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Comparison")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"))
    plt.show()
    print("✅ Results plotted and saved.")


if __name__ == "__main__":
    # Configure matplotlib and numpy
    plt.rcParams["font.size"] = 14
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["axes.grid"] = True
    np.set_printoptions(suppress=True, precision=5)
    warnings.filterwarnings("ignore", category=UserWarning)

    train_loader, test_loader = setup_data()
    criterion = nn.CrossEntropyLoss()
    is_server = input("Run as server? (y/n): ").strip().lower() == "y"

    # Single mode training
    if is_server:
        print("🚀 Starting single mode training...")
        net_single = Net(28 * 28, CONFIG["n_output"], CONFIG["n_hidden"]).to(device)
        optimizer_single = torch.optim.SGD(net_single.parameters(), lr=CONFIG["lr"])
        single_history = train_model(
            net_single,
            train_loader,
            test_loader,
            optimizer_single,
            criterion,
            CONFIG["n_epochs"],
            os.path.join(CONFIG["models_dir"], "single"),
        )

    # Multi mode training
    print("🚀 Starting multi mode training...")
    net_multi = Net(28 * 28, CONFIG["n_output"], CONFIG["n_hidden"]).to(device)
    optimizer_multi = torch.optim.SGD(net_multi.parameters(), lr=CONFIG["lr"])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if is_server:
        sock.bind((CONFIG["host"], CONFIG["port"]))
        sock.listen(1)
        print(
            f"🌐 Server is waiting for a connection on {CONFIG['host']}:{CONFIG['port']}..."
        )
        conn, _ = sock.accept()
        print("🔗 Connection established with client.")
    else:
        while True:
            try:
                print(
                    f"🌐 Connecting to server at {CONFIG['host']}:{CONFIG['port']}..."
                )
                sock.connect((CONFIG["host"], CONFIG["port"]))
                conn = sock
                print("🔗 Connected to server.")
                break
            except ConnectionRefusedError:
                print("❌ Connection failed. Retrying in 5 seconds...")
                import time

                time.sleep(5)

    multi_history = train_model(
        net_multi,
        train_loader,
        test_loader,
        optimizer_multi,
        criterion,
        CONFIG["n_epochs"],
        os.path.join(CONFIG["models_dir"], "multi"),
        is_multi=True,
        conn=conn,
        role="server" if is_server else "client",
    )

    conn.close()
    sock.close()

    # Plot results
    if is_server:
        plot_results(single_history, multi_history, CONFIG["outs_dir"])
