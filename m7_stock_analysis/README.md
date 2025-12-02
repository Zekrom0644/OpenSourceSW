# M7 Stock Analysis System: Hybrid LSTM & Transformer 📈
### Spatio-Temporal Analysis of "Magnificent 7" Stocks

## 📖 Project Overview
본 프로젝트는 미국 주식 시장을 주도하는 **M7(Magnificent 7)** 종목들의 주가 흐름을 **시공간적(Spatio-Temporal) 관점**에서 분석하고 예측하는 하이브리드 모델링 시스템입니다.

단순한 시계열 예측의 한계를 극복하기 위해 두 가지 모델을 결합하여 시장을 입체적으로 분석합니다.
1.  **Temporal Analysis (LSTM):** 개별 종목의 시간적 흐름과 추세(Trend)를 예측
2.  **Spatial Analysis (Transformer):** 종목 간의 영향력과 동조화(Coupling) 현상을 분석

> *"LSTM으로 나무(개별 추세)를 보고, Transformer로 숲(시장 맥락)을 읽는다."*

---

## 🛠 Methodology & Key Features

### 1. Data Engineering (Stationarity)
- **Problem:** 주가(Raw Price)는 비정상성(Non-stationary) 데이터로, 학습 시 단순 평균값 회귀(Mean Prediction) 문제가 발생함.
- **Solution:** **로그 수익률(Log Returns)**로 변환하여 정상성(Stationarity)을 확보하고, 등락 패턴(Pattern)을 학습하도록 개선.
- **Reconstruction:** 예측된 수익률을 다시 주가($)로 변환하여 직관적인 결과 제공.

### 2. LSTM (Temporal Prediction)
- 과거 30일간의 데이터를 입력받아 **단기 추세(Trend)**를 예측.
- **Result:** 하락장(Downturn)이나 변동성 장세에서도 실제 주가의 방향성을 정확히 추종.

### 3. Transformer (Spatial Influence)
- **Self-Attention** 메커니즘을 활용해 종목 간의 **민감도(Sensitivity)** 분석.
- **Directional Heatmap:**
    - 단순 상관계수(대칭)가 아닌, **인과관계(비대칭)**를 분석.
    - **Source(X축) -> Target(Y축)** 형태의 영향력 지도 생성.
    - **Leader vs Follower:** 시장을 주도하는 대장주(예: NVDA)와 추종주를 식별.

---

## 📂 Project Structure

데이터 수집부터 분석까지 5단계의 파이프라인으로 구성되어 있습니다.

| Step | File Name | Description |
|:---:|:---|:---|
| **01** | `get_data.py` | Yahoo Finance에서 M7 데이터 수집 및 전처리 |
| **02** | `preprocess_lstm.py` | LSTM 학습용 데이터셋 생성 (Log Returns 변환) |
| **03** | `train_lstm.py` | LSTM 학습, **주가 복원(Reconstruction)** 및 예측 그래프 저장 |
| **04** | `preprocess_transformer.py` | Transformer 분석용 데이터셋 생성 |
| **05** | `train_transformer.py` | Transformer 학습 및 **Directional Heatmap** 저장 (민감도 분석) |

---

## 💻 Installation & Usage

### 1. Environment Setup
`conda`를 사용하여 가상환경을 설정합니다.

```bash
# 가상환경 생성 (Python 3.10)
conda create -n m7_analysis python=3.10 -y

# 가상환경 활성화
conda activate m7_analysis

# 필수 라이브러리 설치
pip install torch torchvision torchaudio pandas numpy scikit-learn matplotlib seaborn yfinance
