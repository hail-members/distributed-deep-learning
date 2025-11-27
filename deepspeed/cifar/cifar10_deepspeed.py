import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import deepspeed
import argparse

# 1. 모델 정의 (간단한 CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # CrossEntropyLoss를 위해 Logits 반환
        return x

def main():
    # 2. Argument Parser 설정
    parser = argparse.ArgumentParser(description='CIFAR-10 DeepSpeed Example')
    
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    
    # 2-1. (중요) DeepSpeed가 주입하는 --local_rank 인자 받기
    #      이전 에러를 방지하기 위해 공식 튜토리얼에도 이 코드가 포함되어 있습니다.
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    
    # # deepspeed 의 config 파일 이름 받기
    # parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
    #                     help='DeepSpeed config file path')
    
    # 2-2. DeepSpeed의 나머지 인자들 추가
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()

    # 3. 데이터 로더 준비
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    # 4. 모델 및 손실 함수
    net = Net()
    criterion = nn.CrossEntropyLoss()
    
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=net.parameters(),
            training_data=trainset  # ⬅️ 데이터셋을 여기서 전달
            # config="ds_config.json", 이렇게해도되지만 명령어에서 직접 넣어준다면 굳이
            # 참고: train_batch_size는 ds_config.json의
            # "train_micro_batch_size_per_gpu" 값을 자동으로 사용합니다.
    )

    print(f"Starting training on device: {model_engine.device}")

    # 6. 학습 루프
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # inputs과 labels를 모델 엔진의 디바이스로 이동
            inputs, labels = data
            inputs = inputs.to(model_engine.device)
            labels = labels.to(model_engine.device)
            
            # (중요) DeepSpeed 설정에 맞춰 데이터 타입 변환
            if model_engine.fp16_enabled():
                inputs = inputs.half()
            elif model_engine.bfloat16_enabled():
                inputs = inputs.to(torch.bfloat16)

            # Forward pass
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            # DeepSpeed Backward/Step
            model_engine.backward(loss)
            model_engine.step()

            # 통계 출력 (global rank 0에서만)
            if model_engine.global_rank == 0:
                running_loss += loss.item()
                if i % 100 == 99: # 100 mini-batches마다 출력
                    print(f'[rank{model_engine.global_rank} {epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    main()