import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# DDP 설정 함수 (수정 없음)
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12555"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

# MLP 모델 정의 (수정 없음)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Trainer 클래스 (수정 없음)
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        rank: int,
        save_every: int,
    ):
        self.rank = rank
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.model = DDP(self.model)

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.rank == 0 and (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

# 데이터 로더 준비 함수 (수정 없음)
def prepare_dataloader(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

# --- 여기가 핵심적인 수정 부분 ---
def main(total_epochs: int, save_every: int, batch_size: int):
    # torchrun이 설정한 환경 변수에서 rank와 world_size를 읽어옵니다.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    ddp_setup(rank, world_size)
    
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_data = prepare_dataloader(batch_size)
    
    print(f"Rank {rank} 훈련 시작")
    
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DDP MNIST CPU Example (torchrun)')
    parser.add_argument('--total_epochs', type=int, default=5, help='Total epochs to train.')
    parser.add_argument('--save_every', type=int, default=2, help='How often to save a checkpoint.')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training.')
    args = parser.parse_args()
    
    main(args.total_epochs, args.save_every, args.batch_size)