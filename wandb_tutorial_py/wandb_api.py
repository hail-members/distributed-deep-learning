"""
Wandb.Api를 사용하여 이전 run들의 기록을 가져와
Matplotlib으로 비교 그래프를 그리는 스크립트입니다.

실행 전 'YOUR_USERNAME'과 'YOUR_PROJECT_NAME'을
본인의 정보로 수정해야 합니다.

1. pip install matplotlib pandas
2. (wandb login은 이미 되어있다고 가정)
3. python plot_wandb_runs.py
"""

import wandb
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 설정: 사용자 이름과 프로젝트 이름을 입력하세요 ---
# wandb.ai/YOUR_USERNAME/YOUR_PROJECT_NAME 에서 확인
# 또는 wandb_mlp_tutorial.py 실행 시 'entity'와 'project'에서 확인
# 예: YOUR_USERNAME = "my-username"
YOUR_USERNAME = "dongjaekim" # <-- 본인의 wandb 사용자 이름(entity)으로 변경
YOUR_PROJECT_NAME = "mlp-wandb-tutorial-py" # <-- 이전 스크립트의 project 이름
# -----------------------------------------------------

# 1. wandb.Api 초기화
api = wandb.Api()

# 2. 프로젝트의 모든 run 가져오기
runs_path = f"{YOUR_USERNAME}/{YOUR_PROJECT_NAME}"
try:
    runs = api.runs(path=runs_path)
except Exception as e:
    print(f"'{runs_path}'에서 run을 가져오는 데 실패했습니다: {e}")
    print("스크립트 상단의 'YOUR_USERNAME'과 'YOUR_PROJECT_NAME'이 올바르게 설정되었는지 확인하세요.")
    print("wandb.ai 에서 로그인 후 본인의 사용자 이름(entity)을 확인해야 합니다.")
    exit()

print(f"'{runs_path}' 프로젝트에서 {len(runs)}개의 run을 찾았습니다.")

# 3. Matplotlib 시각화 준비 (예쁘게)
# 'ggplot' 스타일을 사용하면 깔끔한 그래프를 그릴 수 있습니다.
plt.style.use('ggplot')
plt.figure(figsize=(12, 7))

# 4. 각 run을 순회하며 데이터 플로팅
for run in runs:
    # 5. run의 기록(history) 가져오기
    # history()는 Pandas DataFrame을 반환합니다.
    # 'Validation/Accuracy'가 없는 step(NaN)을 제거하기 위해 dropna() 사용
    try:
        # 필요한 key만 가져오고, 'Validation/Accuracy'가 NaN인 행(step)은 제거
        history = run.history(keys=['_step', 'Validation/Accuracy']).dropna()
        
        if history.empty:
            print(f"Run '{run.name}'에는 'Validation/Accuracy' 데이터가 없습니다. 건너뜁니다.")
            continue

        # 6. Matplotlib으로 플로팅
        # label에 run의 이름과 ID 앞 4자리를 넣어 구분
        plt.plot(
            history['_step'], 
            history['Validation/Accuracy'], 
            label=f"{run.name} (ID: {run.id[:4]})" 
        )
    except Exception as e:
        print(f"Run '{run.name}' 플로팅 중 오류 발생: {e}")

# 7. 그래프 꾸미기 (라벨, 제목, 범례)
plt.title(f"'{YOUR_PROJECT_NAME}' 프로젝트: Validation Accuracy 비교")
plt.xlabel('Global Steps')
plt.ylabel('Validation Accuracy (%)')
plt.legend() # 범례 표시
plt.grid(True) # 그리드 표시
plt.tight_layout() # 레이아웃을 깔끔하게

# 8. 그래프 저장 및/또는 표시
output_filename = "wandb_accuracy_comparison.png"
plt.savefig(output_filename)
print(f"그래프가 '{output_filename}' 파일로 저장되었습니다.")

# 로컬 환경에서 바로 차트를 보려면 주석 해제
# plt.show()