# 🎬 MovieFinder: AI Semantic Search
---

  

## 1. 프로젝트 개요 (Project Overview)

**MovieFinder**는 사용자가 기억나는 영화의 특정 장면이나 분위기를 문장으로 묘사하면, AI가 수만 개의 영화 줄거리 데이터를 분석하여 가장 유사한 영화를 찾아주는 **지능형 검색 프로그램**입니다.

단순 키워드 매칭 방식과 달리, **BERT 기반의 언어 모델**을 도입하여 문맥을 이해하며, 검색 결과는 **OpenCV**를 활용한 GUI 카드로 시각화하여 제공합니다.

  

###  핵심 기능

- **Semantic Search:** "침몰하는 배와 사랑 이야기"라고 검색해도 줄거리에 해당 단어가 없지만 문맥이 일치하는 "Titanic"을 찾아냅니다.

- **Vector Indexing:** 방대한 영화 데이터를 미리 벡터화하여 저장(`movie_index.pt`)함으로써 실시간 검색 속도를 최적화했습니다.

- **Visual Interface:** 터미널 결과뿐만 아니라, OpenCV를 활용해 영화 제목, 연도, 유사도, 줄거리를 시각적인 카드로 보여줍니다.

  

---

  

## 2. 데모 및 실행 예시 (Demo)

사용자가 **"A mouse cooking in a restaurant"** 라고 입력했을 때, AI가 이를 분석하여 **Ratatouille**를 추천하는 모습입니다.

  
![](https://raw.githubusercontent.com/seoj00/pic/refs/heads/main/demo1.png)
![](https://raw.githubusercontent.com/seoj00/pic/refs/heads/main/demo2.png)
![](https://raw.githubusercontent.com/seoj00/pic/refs/heads/main/demo3.png)


---

  

## 3. 사용한 패키지 및 버전 (Requirements)

본 프로젝트는 **Python 3.8 이상** 환경에서 개발되었습니다. 실행을 위해 아래 라이브러리 설치가 필요합니다.

| Package | Version | Description |
| :--- | :--- | :--- |
| **pandas** | `2.0.0+` | 대용량 CSV 데이터 로드 및 전처리 |
| **sentence-transformers** | `2.2.2+` | 문장 임베딩(Vectorization) 및 AI 모델 로드 |
| **torch** | `2.0.0+` | 텐서 연산 및 딥러닝 프레임워크 |
| **opencv-python** | `4.8.0+` | 결과 시각화 (GUI 창 생성) |
| **numpy** | `1.24.0+` | 수치 연산 보조 |

### 설치 방법 (Installation)
터미널(CMD/Powershell)에서 아래 명령어를 입력하여 필수 패키지를 설치하세요.

```bash
pip install pandas sentence-transformers torch opencv-python numpy
```

---

## 4. 실행 방법 (Usage)

이 프로그램은 **[Step 1: 인덱스 생성]**과 **[Step 2: 검색 실행]** 두 단계로 이루어져 있습니다.

  

### Step 0: 사전 준비 (Directory Check)

실행 전, 파일과 폴더가 아래 구조대로 배치되어 있는지 반드시 확인해주세요. (`modules` 폴더 필수)

```text

MovieFinder/
├── modules/               # [필수] 커스텀 패키지 폴더
│   ├── __init__.py        # (빈 파일)
│   ├── embedder.py        # AI 모델 로딩 및 임베딩 모듈
│   └── ui.py              # OpenCV 결과창 출력 모듈
│
├── 01_build_index.py      # [Step 1] 인덱스 빌드 실행 파일
├── 02_main.py             # [Step 2] 메인 검색 실행 파일
└── imdb_movies.csv        # 영화 데이터셋

```

  

###  Step 1: 검색 인덱스 생성 (Indexing)

영화 줄거리 데이터를 AI가 이해할 수 있는 벡터로 변환하여 저장합니다. (최초 1회 실행)

```bash

python 01_build_index.py

```

- **기능:** `imdb_movies.csv`를 읽어들여 AI 분석 후, `movie_index.pt` 파일을 생성합니다.

- **소요 시간:** 데이터 양(약 1만 개 기준)에 따라 1\~3분 정도 소요될 수 있습니다.

  

###  Step 2: 검색기 실행 (Search)

생성된 인덱스 파일을 기반으로 검색 프로그램을 실행합니다.

```bash

python 02_main.py

```
  

**[사용 가이드]**

1.  **입력:** 터미널 프롬프트에 찾고 싶은 장면을 **영어 문장**으로 묘사합니다.
	- *Example:* "A man fighting in a ring to prove himself"

2.  **결과 확인:** 가장 유사한 영화 정보가 담긴 **OpenCV 윈도우 창**이 팝업됩니다.

3.  **창 닫기:** 키보드의 아무 키나 누르면 창이 닫히고 다음 검색이 가능합니다.

4.  **종료:** `q`를 입력하면 프로그램이 종료됩니다.

  

-----

  

## 5. 참고자료 (References)

본 프로젝트 개발에 참고한 데이터셋과 기술 문서는 다음과 같습니다.

  

###  데이터셋 (Dataset)

- **IMDb Movies Dataset:** [Kaggle - IMDb Movies Dataset](https://www.google.com/search?q=https://www.kaggle.com/ashpalsingh1525/imdb-movies-dataset)

###  AI 모델 및 알고리즘 (Model & Algorithm)

- **Sentence-BERT:** [Hugging Face - Sentence Transformers](https://www.sbert.net/)
    - 사용 모델: `all-MiniLM-L6-v2` (Sentence Embeddings for Semantic Search)

- **Cosine Similarity:** PyTorch `util.cos_sim` 함수를 이용한 고속 유사도 계산

###  라이브러리 문서 (Documentation)

- **OpenCV:** [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) (GUI 구현)

- **Pandas:** [Pandas Documentation](https://pandas.pydata.org/docs/) (데이터 전처리)

- **PyTorch:** [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) (텐서 연산)