import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # CSV 읽기용 추가

# -------------------------------------------
# 1. 설정
# -------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

INPUT_DIM = 7
HIDDEN_DIM = 64
OUTPUT_DIM = 7
NUM_LAYERS = 2
NUM_EPOCHS = 100
LR = 0.001
SEQ_LENGTH = 30 # 복원을 위해 필요

# -------------------------------------------
# 2. 데이터 로드
# -------------------------------------------
try:
    # (1) 학습용 데이터(Tensor) 로드
    data = torch.load('m7_lstm_data.pt', map_location=device, weights_only=False)
    X_train = data['X_train'].to(device)
    y_train = data['y_train'].to(device)
    X_test = data['X_test'].to(device)
    scaler = data['scaler'] # 스케일러 가져오기 (필수)
    
    # (2) 원본 주가 데이터(CSV) 로드 (가격 복원을 위해 필요)
    df = pd.read_csv('m7_stock_data.csv')
    real_prices = df.drop(columns=['Date']).values 
    
    # 테스트 데이터가 시작되는 시점 찾기
    train_size = int(len(data['X_train'])) # 학습 데이터 개수
    test_start_idx = train_size + SEQ_LENGTH # 실제 가격 배열에서의 인덱스

except FileNotFoundError:
    print("오류: 필요한 파일('m7_lstm_data.pt' 또는 'm7_stock_data.csv')이 없습니다.")
    exit()

# -------------------------------------------
# 3. 모델 정의
# -------------------------------------------
class M7LSTM(nn.Module):
    def __init__(self):
        super(M7LSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM).to(device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_DIM).to(device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

model = M7LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# -------------------------------------------
# 4. 학습 루프
# -------------------------------------------
loss_history = []
print("LSTM 학습 시작...")
model.train()

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'm7_lstm_model.pth')
print(">> 모델 저장 완료")

# -------------------------------------------
# 5. 결과 시각화 (Loss & 주가 복원 그래프)
# -------------------------------------------

# (1) Loss 그래프 저장
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='Training Loss')
plt.title("Training Process (Loss)")
plt.savefig('result_lstm_loss.png')
plt.close()

# (2) 주가 복원 및 예측 결과 저장 (핵심!)
print("주가 복원(Reconstruction) 및 결과 저장 중...")
model.eval()
with torch.no_grad():
    # 모델 예측 (값: -1 ~ 1 사이의 스케일링된 수익률)
    pred_scaled = model(X_test).cpu().numpy()
    
    # 정답 데이터 (값: -1 ~ 1 사이의 스케일링된 수익률)
    # y_test는 GPU에 있으므로 cpu로 가져와야 함
    real_scaled = data['y_test'].cpu().numpy()

# A. 스케일링 원복 (수익률 %로 변환)
pred_returns = scaler.inverse_transform(pred_scaled)
real_returns = scaler.inverse_transform(real_scaled)

# B. 가격 재구성 (어제가격 * exp(수익률))
restored_pred = []
restored_real = []

# 테스트 시작 전날의 가격을 기준점으로 잡음
current_pred = real_prices[test_start_idx - 1]
current_real = real_prices[test_start_idx - 1]

for i in range(len(pred_returns)):
    next_pred = current_pred * np.exp(pred_returns[i])
    next_real = current_real * np.exp(real_returns[i])
    
    restored_pred.append(next_pred)
    restored_real.append(next_real)
    
    current_pred = next_pred
    current_real = next_real

restored_pred = np.array(restored_pred)
restored_real = np.array(restored_real)

# 그래프 그리기 (첫 번째 종목: AAPL)
target_idx = 0 
plt.figure(figsize=(12, 6))
plt.plot(restored_real[:, target_idx], label='Actual Price', color='blue', alpha=0.6)
plt.plot(restored_pred[:, target_idx], label='AI Predicted Price', color='red', linestyle='--')
plt.title("M7 Stock Price Prediction (Reconstructed)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.savefig('result_lstm_prediction.png')
plt.close()

print(">> 완료! 'result_lstm_prediction.png' 파일을 확인해보세요. (진짜 주가 그래프)")
