import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# --- 1. 모델 정의 ---
# 'hidden_size'를 wandb.config에서 받아옵니다.
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # (N, 1, 28, 28) -> (N, 784)
        x = x.view(x.shape[0], -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # CrossEntropyLoss는 softmax를 포함하므로 여기서는 log_softmax 불필요
        return x

# --- 2. 데이터 로더 준비 ---
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# --- 3. 학습 및 평가 로직 ---
def train_and_evaluate():
    # ⭐ (중요) wandb.init() 호출
    # Sweep Controller가 이 스크립트를 실행할 때 config 값을 주입합니다.
    run = wandb.init(
        name = "mlp-sweep-run",  # 각 실행(run)의 이름
        )
    
    # ⭐ (중요) Sweep에서 정의한 하이퍼파라미터 접근
    # config의 기본값(defaults)을 설정할 수도 있습니다.
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 데이터 로드
    train_loader, test_loader = get_data_loaders()
    
    # 2. 모델 생성 (config 값 사용)
    model = SimpleMLP(
        hidden_size=config.hidden_size
    ).to(device)
    
    # 3. 옵티마이저 생성 (config 값 사용)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 4. 학습 루프 (config 값 사용)
    for epoch in range(config.epochs):
        # Train
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Test (매 에포크마다)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0) # 배치 로스 합
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f"Epoch {epoch}: Test Loss={test_loss:.4f}, Acc={test_accuracy:.2f}%")
        
        # 📈 (중요) wandb에 Metric 로깅
        # Sweep은 이 값('test_loss')을 기준으로 최적화를 수행합니다.
        wandb.log({
            "epoch": epoch,
            "Test/test loss": test_loss,
            "test_accuracy": test_accuracy
        })

if __name__ == "__main__":
    train_and_evaluate()