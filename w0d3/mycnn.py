import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from PIL import Image
import PIL
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import nn
import torch as t
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from torchvision import datasets, transforms

import w0d3.utils
from w0d2.my_modules import Conv2d, Flatten, Linear, MaxPool2d, ReLU

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels=1,
            out_channels=5,
            kernel_size=(3,3),
            stride=1,
            padding=0)
        self.relu2 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv4 = Conv2d(
            in_channels=5,
            out_channels=8,
            kernel_size=(3,3),
            stride=1,
            padding=1)
        self.maxpool5 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.flatten6 = Flatten()
        self.linear7 = Linear(3136, 128)
        self.linear8 = Linear(128, 10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x2 = self.conv1(x)
        x3 = self.relu2(x2)
        x4 = self.maxpool3(x3)
        x5 = self.conv4(x4)
        x6 = self.maxpool5(x5)
        x7 = self.flatten6(x6)
        x8 = self.linear7(x7)
        x9 = self.linear8(x8)
        return x9


epochs = 3
loss_fn = nn.CrossEntropyLoss()
batch_size = 128

MODEL_FILENAME = "./w1d2_convnet_mnist.pt"
device = "cuda" if t.cuda.is_available() else "cpu"

def train_convnet(trainloader: DataLoader, epochs: int, loss_fn: Callable) -> list:
    '''
    Defines a ConvNet using our previous code, and trains it on the data in trainloader.
    '''

    model = ConvNet().to(device).train()
    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

    print(f"Saving model to: {MODEL_FILENAME}")
    t.save(model, MODEL_FILENAME)
    return loss_list

if __name__ == '__main__':
    model = ConvNet()
    # print(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')


    # TODO. Into Jupyter:
    loss_list = train_convnet(trainloader, epochs, loss_fn)

    fig = px.line(y=loss_list, template="simple_white")
    fig.update_layout(title="Cross entropy loss on MNIST", yaxis_range=[0, max(loss_list)])
    fig.show()
