import torch 
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        prob = self.linear_relu_stack(x)
        return prob

def prepare_data(batch_size=32):

    trainset = torchvision.datasets.MNIST(
                            root="data",                                        # path to where data is stored
                            train=True,                                         # specifies if data is train or test
                            download=True,                                      # downloads data if not available at root
                            transform=torchvision.transforms.ToTensor()         # trasforms both features and targets accordingly
                            )
    
    # pass data to the distributed sampler and dataloader 
    train_dataloader = DataLoader(trainset,
                                  shuffle=False,
                                  batch_size=batch_size)
    
    return train_dataloader

# training loop for one epoch
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # transfer data to GPU if available
        X = X.cuda()
        y = y.cuda()
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    
    ################################################
    # 1. Setup Dataloader 
    train_dataloader = prepare_data()
    ################################################

    ################################################                                                 
    # 2. Set the model
    model = Net().cuda()
    ################################################
    
    # instantiate loss and optimizer 
    loss_fn = torch.nn.CrossEntropyLoss() #torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model 
    epochs = 10
    for t in range(epochs):
        train_loop(train_dataloader, model, loss_fn, optimizer)

    print("Done!")
    return model

if __name__ == "__main__":
    world_size= torch.cuda.device_count()
    print('world_size = {}'.format(world_size))
    main()
#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#  world_size=1
#  main(device, world_size) 
