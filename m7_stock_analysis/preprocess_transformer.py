import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# 1. 설정
SEQ_LENGTH = 30 
FILE_PATH = 'm7_stock_data.csv'

# 2. 데이터 로드 및 수익률 변환 (LSTM과 동일 로직)
df = pd.read_csv(FILE_PATH, index_col=0)
returns = np.log(df.values[1:] / df.values[:-1]) 

# 3. 스케일링
scaler = MinMaxScaler(feature_range=(-1, 1))
returns_scaled = scaler.fit_transform(returns)

# 4. 윈도우 생성
X, y = [], []
for i in range(len(returns_scaled) - SEQ_LENGTH):
    X.append(returns_scaled[i : i + SEQ_LENGTH])
    y.append(returns_scaled[i + SEQ_LENGTH])

# 5. 저장 (이름만 다르게)
X_tensor = torch.FloatTensor(np.array(X))
y_tensor = torch.FloatTensor(np.array(y))
train_size = int(len(X_tensor) * 0.8)

torch.save({
    'X_train': X_tensor[:train_size],
    'y_train': y_tensor[:train_size],
    'X_test': X_tensor[train_size:],
    'scaler': scaler,
    'column_names': df.columns.tolist() # 히트맵 그릴 때 종목 이름 필요
}, 'm7_transformer_data.pt')

print("Transformer용 전처리 완료: m7_transformer_data.pt 생성됨")
