"""
PyTorch와 Wandb(Weights & Biases)를 사용한 MLP 튜토리얼

이 스크립트는 다음 작업을 수행합니다:
1. MNIST 데이터셋으로 간단한 MLP(다층 퍼셉트론) 모델을 학습합니다.
2. Wandb(Weights & Biases)를 사용하여 세 가지 주요 항목을 로깅합니다:
    - Scalars: 훈련 손실(step별), 검증 손실 및 정확도(epoch별)
    - Image: 테스트 데이터셋의 이미지 샘플
    - Graph: MLP 모델의 아키텍처, 파라미터, 그래디언트
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb # Wandb 임포트
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.03)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--runtype', type=str, default='MainExp')

args = parser.parse_args()
args.lr
args.batch_size
args.epochs
args.runtype

# --- 1. 설정 및 하이퍼파라미터 ---

# 하이퍼파라미터
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# --- 2. Wandb 초기화 ---
# wandb.init()을 호출하여 새 run을 시작합니다.
# 이 함수가 실행되면 터미널에 대시보드 URL이 출력됩니다.

# run = wandb.init(
#     project="mlp-wandb-tutorial-py", # 프로젝트 이름 (자유롭게 설정)
#     entity=None,                     # 개인 계정(None) 또는 팀 이름
#     config={                         # 하이퍼파라미터를 config에 저장
#         "learning_rate": learning_rate,
#         "batch_size": batch_size,
#         "epochs": num_epochs,
#         "architecture": "Simple MLP",
#         "dataset": "MNIST"
#     },
#     name=args.runtype # 지정하지 않으면 랜덤한 이름으로 나온다
# )

# 또는

run = wandb.init(
    project="mlp-wandb-tutorial-py", # 프로젝트 이름 (자유롭게 설정)
    entity="dongjaekim",  # "hails" # 개인 계정(저의경우 "dongjaekim") 또는 팀 이름
    config=args.__dict__, # 항상 이렇게 써야하는건 아니고 이렇게 쓰는 경우가 많아서 이렇게 보여드리는거
    name=args.runtype
)

# config 객체를 통해 하이퍼파라미터에 접근 (wandb.config)
config = wandb.config
print(config)



# --- 3. 데이터 준비 (MNIST) ---

# 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize((0.5,), (0.5,))  # -1 ~ 1 범위로 정규화
])

# 학습 데이터셋
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size, # config에서 하이퍼파라미터 참조
    shuffle=True
)

# 테스트 데이터셋
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    shuffle=False
)

# --- 4. 모델 정의 (간단한 MLP) ---

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 28*28 (784) 입력
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10개 클래스 (0~9) 출력
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

model = MLP().to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)


# --- 5. Wandb 로깅 ---

# 3: 모델 그래프(Graph) 및 그래디언트 저장
# wandb.watch()는 모델을 관찰하며 그래디언트, 파라미터, 모델 구조를 로깅합니다.
# (TensorBoard의 add_graph와 유사하지만, 그래디언트까지 추적합니다)
wandb.watch(model, criterion, log="all", log_freq=1)
print("Wandb: 모델 그래프(watch) 설정 완료.")

# 테스트 로더에서 이미지 배치를 하나 가져옵니다.
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 2: 테스트 데이터 이미지 샘플 저장
# wandb.Image()를 사용하여 이미지 리스트를 로깅합니다.
# 캡션(caption)을 추가하여 라벨을 표시할 수 있습니다.
wandb.log({"Test/Image_Samples": [
    wandb.Image(img, caption=f"Label: {label}")
    for img, label in zip(images[:16], labels[:16])
]})
print("Wandb: 이미지 샘플 저장 완료.")


# --- 6. 모델 학습 및 검증 루프 ---

print("모델 학습을 시작합니다...")
global_step = 0
for epoch in range(config.epochs):
    model.train()  # 학습 모드
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 100 스텝마다 훈련 손실(loss) 로깅
        if (i + 1) % 100 == 0:
            # 1 (Scalar): 훈련 손실 저장
            # wandb.log() 함수 하나로 모든 것을 로깅합니다.
            wandb.log({
                'Training/Loss': running_loss / 100,
            }, step=global_step + i) # 스텝(x축)을 명시
            running_loss = 0.0 # 리셋

    global_step += len(train_loader) # epoch이 끝날 때 global_step 업데이트

    # --- 매 Epoch 종료 시 검증 ---
    model.eval()  # 평가 모드
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 1 (Scalar): 검증 손실 및 정확도 저장 (Epoch 기준)
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 딕셔너리 형태로 한 번에 여러 값을 로깅
    wandb.log({
        'Validation/Loss': avg_test_loss,
        'Validation/Accuracy': accuracy
    }, step=global_step) # Epoch이 아닌 global_step 기준으로 x축 통일

    print(f'Epoch [{epoch+1}/{config.epochs}], '
          f'Val Loss: {avg_test_loss:.4f}, '
          f'Val Accuracy: {accuracy:.2f}%')

# --- 7. 완료 ---
# wandb.finish()를 호출하여 run을 종료합니다.
wandb.finish()
print("학습 완료. Wandb 대시보드에서 결과를 확인하세요.")