import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------------------
# 1. 설정
# -------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

INPUT_DIM = 7
NUM_HEADS = 1 
HIDDEN_DIM = 64
NUM_LAYERS = 1
NUM_EPOCHS = 50

# -------------------------------------------
# 2. 데이터 로드
# -------------------------------------------
try:
    data = torch.load('m7_transformer_data.pt', map_location=device, weights_only=False)
except FileNotFoundError:
    print("오류: 'm7_transformer_data.pt' 파일이 없습니다.")
    exit()

X_train = data['X_train'].to(device)
y_train = data['y_train'].to(device)
col_names = data['column_names'] 

# -------------------------------------------
# 3. 모델 정의
# -------------------------------------------
class M7Transformer(nn.Module):
    def __init__(self):
        super(M7Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=INPUT_DIM, nhead=NUM_HEADS, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(INPUT_DIM * 30, 7) 

    def forward(self, x):
        encoded = self.transformer_encoder(x)
        flatten = encoded.view(encoded.size(0), -1) 
        return self.fc(flatten)

model = M7Transformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# -------------------------------------------
# 4. 학습 루프
# -------------------------------------------
print("Transformer 학습 시작...")
model.train()
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

print("학습 완료!")

# -------------------------------------------
# 5. [축 변경됨] 방향성 민감도 분석 (Directional Sensitivity)
# X축: Source (원인), Y축: Target (결과)
# -------------------------------------------
print("방향성(Directional) 영향력 분석 중...")
model.eval()

# (1) 기준 데이터 및 예측값
base_input = X_train[-1].unsqueeze(0) 
base_prediction = model(base_input).cpu().detach().numpy()[0] 

# (2) 민감도 행렬 초기화
influence_matrix = np.zeros((7, 7))

# (3) 분석 시작
with torch.no_grad():
    for i in range(7): # i: Source (원인, 변화를 주는 놈) -> X축으로 보냄
        perturbed_input = base_input.clone()
        
        # [핵심] 원인 종목을 '상승(+)' 시킴
        noise = torch.std(perturbed_input[0, :, i]) * 2.0 
        perturbed_input[0, :, i] += noise
        
        new_prediction = model(perturbed_input).cpu().numpy()[0]
        
        # 변화량 계산 (새 예측값 - 기존 예측값)
        diff = new_prediction - base_prediction
        
        for j in range(7): # j: Target (결과, 반응하는 놈) -> Y축으로 보냄
            # 행(Row, Y축)에 Target(j), 열(Col, X축)에 Source(i)를 넣음
            influence_matrix[j, i] = diff[j]

# (4) 정규화
max_val = np.max(np.abs(influence_matrix))
if max_val > 0:
    influence_matrix = influence_matrix / max_val

# (5) 히트맵 그리기
plt.figure(figsize=(10, 8))

# cmap='coolwarm': 파랑(음수) ~ 하양(0) ~ 빨강(양수)
ax = sns.heatmap(influence_matrix, annot=True, fmt=".2f", 
            xticklabels=col_names, yticklabels=col_names, 
            cmap='coolwarm', center=0, vmin=-1, vmax=1) 

# [수정된 부분] 축 레이블 변경
plt.xlabel("Source Stock (Impulse / Changed)", fontsize=12, fontweight='bold')   # X축: 원인
plt.ylabel("Target Stock (Response / Reacted)", fontsize=12, fontweight='bold') # Y축: 결과
plt.title("Directional Influence Map (X -> Y)", fontsize=14)

plt.tight_layout()
plt.show() # 또는 plt.savefig('result_heatmap.png')

print(">> 분석 완료! X축 종목이 변했을 때, Y축 종목이 어떻게 반응하는지 보여줍니다.")
