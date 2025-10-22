import torch.distributed as dist
import torch 
from torch.utils.data import DataLoader
import numpy as np
# import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook, PowerSGDState

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

# Remove rank and world_size -- use what's in 
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
                                  sampler=DistributedSampler(trainset),# num_replicas=world_size, rank=rank),
                                  batch_size=batch_size)

    return train_dataloader


##################################################################################
# 1A. Remove code that sets environment variables as this done for you automatically with torchrun.
def init_distributed():   #rank, world_size):
    # 1B. Instead, use these environment variables set by pytorch and instead of explicitly defining them.
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)                       #
    dist.init_process_group("nccl",                   # backend being used; nccl typically used with distributed GPU training
                            rank=local_rank,                # rank of the current process being used
                            world_size=world_size)    # total number of processors being used
#############################################################

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

        if rank == 0:
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    #####################################################################
    # 1.B We also create the variable local_rank in our main function as well as call the new init_distributed()
    # this will be used to assign the gpu where our model should reside
    local_rank = int(os.environ['LOCAL_RANK'])

    init_distributed()
    ################################################

    train_dataloader = prepare_data()

    ################################################                                                 
    # 2.A. Create location to store checkpoints

    # Create directory for storing checkpointed model
    model_folder_path = os.getcwd()+"/output_model_mnist/"    # create variable for path to folder for checkpoints
    os.makedirs(model_folder_path,exist_ok=True)        # create directory for models if they do not exist
    # create file name for checkpoint 
    checkpoint_file = model_folder_path+"best_model.pt" # create filename for model checkpoint
    ################################################

    # instantiate network and set to local_rank device
    net = Net().to(local_rank)
    
    #################################################
    # 2B. Read checkpoints if they exist 
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(local_rank))
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch']
    
    # otherwise we are starting training from the beginning at epoch 0
    else:
        epoch_start = 0
    ################################################
    
    model = DDP(net,
            device_ids=[local_rank],                  # list of gpu that model lives on 
            output_device=local_rank,                 # where to output model
        )
    powersgd_state = PowerSGDState(
            process_group=model.process_group,  # DDP 모델의 프로세스 그룹 사용
            matrix_approximation_rank=1,
            start_powerSGD_iter=5  # 0 스텝 이후부터 압축 시작 (튜닝 필요)
        )
    model.register_comm_hook(powersgd_state, powerSGD_hook)

    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_every = 1
    epochs = 10
    ###########################################################
    # 2C. Resume training at epoch last checkpoint was written
    for epoch in range(epoch_start, epochs): # note we start loop at epoch_start defined in code above
    ###########################################################
        train_loop(local_rank, train_dataloader, model, loss_fn, optimizer)
        ###########################################################
        # 2D. Write checkpoints periodically during training
        if local_rank == 0 and epoch%save_every==0:
            print(f"Epoch {epoch+1}\n-------------------------------")
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.module.state_dict(),
            }, checkpoint_file)
            print("Finished saving model\n")
    ############################################################

    dist.destroy_process_group()
    
    print("Done!")
    return model

if __name__ == "__main__":
    ############################################################
    # 4. Remove using the mp.spawn to parallelize code and replace this with a function call, as this is done automatically by torchrun
    ############################################################  
    main()

# torchrun --standalone --nnodes=1 --nproc_per_node=2 mnist_powersgd_torchrun.py