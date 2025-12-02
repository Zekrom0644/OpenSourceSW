import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# 1. 설정
SEQ_LENGTH = 30  # 30일치를 보고 내일을 예측
FILE_PATH = 'm7_stock_data.csv'

# 2. 데이터 로드 및 수익률 변환
df = pd.read_csv(FILE_PATH, index_col=0)
prices = df.values
# 로그 수익률 계산: ln(오늘/어제)
returns = np.log(prices[1:] / prices[:-1]) 

# 3. 스케일링 (-1 ~ 1 사이로 변환)
scaler = MinMaxScaler(feature_range=(-1, 1))
returns_scaled = scaler.fit_transform(returns)

# 4. 슬라이딩 윈도우 생성
X, y = [], []
for i in range(len(returns_scaled) - SEQ_LENGTH):
    X.append(returns_scaled[i : i + SEQ_LENGTH]) # 30일치 입력
    y.append(returns_scaled[i + SEQ_LENGTH])     # 다음날 정답

# 5. 텐서 변환 및 저장
X_tensor = torch.FloatTensor(np.array(X))
y_tensor = torch.FloatTensor(np.array(y))

# 학습/테스트 분리 (80%)
train_size = int(len(X_tensor) * 0.8)

torch.save({
    'X_train': X_tensor[:train_size],
    'y_train': y_tensor[:train_size],
    'X_test': X_tensor[train_size:],
    'y_test': y_tensor[train_size:],
    'scaler': scaler
}, 'm7_lstm_data.pt')

print("LSTM용 전처리 완료: m7_lstm_data.pt 생성됨")
