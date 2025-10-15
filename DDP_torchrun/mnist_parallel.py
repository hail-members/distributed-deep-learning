import torch.distributed as dist
import torch 
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
from torch.utils.data.distributed import DistributedSampler
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

def prepare_data(rank, world_size, batch_size=32):

    trainset = torchvision.datasets.MNIST(
                            root="data",                                        # path to where data is stored
                            train=True,                                         # specifies if data is train or test
                            download=True,                                      # downloads data if not available at root
                            transform=torchvision.transforms.ToTensor()         # trasforms both features and targets accordingly
                            )
    
    # pass data to the distributed sampler and dataloader 
    train_dataloader = DataLoader(trainset,
                                  shuffle=False,
                                  sampler=DistributedSampler(trainset, num_replicas=world_size, rank=rank),
                                  batch_size=batch_size)
    
    return train_dataloader

def init_distributed(rank, world_size):
    '''
    rank: identifier for pariticular GPU
    world: total number of process in a the group
    '''
    os.environ['MASTER_ADDR'] = 'localhost'           # IP address of rank 0 process
    os.environ['MASTER_PORT'] = '12355'               # a free port used to communicate amongst processors
    torch.cuda.set_device(rank)                       #
    dist.init_process_group("gloo",       #"nccl",                   # backend being used; nccl typically used with distributed GPU training
                            rank=rank,                # rank of the current process being used
                            world_size=world_size)    # total number of processors being used

from torch.nn.parallel import DistributedDataParallel as DDP

# training loop for one epoch
def train_loop(rank, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # transfer data to GPU if available
        X = X.to(rank)
        y = y.to(rank)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ################################################
#        # 4. Only write/print model information on one GPU
        if rank == 0:
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        ################################################

def main(rank, world_size):
    ################################################
    # 1. Set up Process Group 
    init_distributed(rank, world_size)
    ################################################

    ################################################
    # 2. Setup Dataloader with Distributed Sampler
    train_dataloader = prepare_data(rank, world_size)
    ################################################

    ################################################                                                 
    # 3. Wrap Model with DDP  
#    model = Net()
    model = DDP(Net().to(rank),
        device_ids=[rank],                  # list of gpu that model lives on 
        output_device=rank,                 # where to output model
    )        
    ################################################
    
    # instantiate loss and optimizer 
    loss_fn = torch.nn.CrossEntropyLoss() #torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model 
    epochs = 10
    for t in range(epochs):
        ################################################
        # 4. Only write/print model information on one GPU
        if rank == 0:
            print(f"Epoch {t+1}\n-------------------------------")
        ################################################
        train_loop(rank, train_dataloader, model, loss_fn, optimizer)

    #################################################
    # 5. Close Process Group
    dist.destroy_process_group()
    #################################################
    print("Done!")
    return model

if __name__ == "__main__":
    world_size= torch.cuda.device_count()
    print('world_size = {}'.format(world_size))
    mp.spawn(main, args=(world_size,) , nprocs=world_size)
#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#  world_size=1
#  main(device, world_size) 
