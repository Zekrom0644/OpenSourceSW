import yfinance as yf
import pandas as pd

# 1. 설정
tickers = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
start_date = '2020-01-01'
end_date = '2024-12-31' # 오늘 날짜로 변경 가능

print("데이터 수집을 시작합니다...")

# 2. 데이터 다운로드 (종가 기준)
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# 3. 결측치 처리 (ffill)
data = data.ffill().dropna()

# 4. 저장
data.to_csv('m7_stock_data.csv')
print(f"다운로드 완료! 데이터 크기: {data.shape}")
print("파일 저장 완료: m7_stock_data.csv")
