import pandas as pd
import torch
import os
from modules.embedder import MovieSearchEngine

CSV_PATH = 'imdb_movies.csv'  # 다운받은 IMDb 파일 이름
SAVE_PATH = 'movie_index.pt'
LIMIT_DATA = 10000            # 데이터가 많으므로 1만 개 정도로 설정

def main():
    print("=========================================")
    print("      🎬 MovieFinder Index Builder      ")
    print("=========================================")

    # 1. CSV 데이터 로드
    if not os.path.exists(CSV_PATH):
        print(f"[Error] '{CSV_PATH}' 파일을 찾을 수 없습니다.")
        print("프로젝트 폴더 안에 csv 파일이 있는지 확인해주세요.")
        return

    print(f"[Step 1] 데이터 로딩 중... ({CSV_PATH})")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[Error] CSV 파일을 읽는 도중 오류 발생: {e}")
        return
    
    df = df.rename(columns={
        'names': 'Title',        # 제목
        'overview': 'Plot',      # 줄거리
        'date_x': 'Release Year' # 개봉일
    })
    
    # 데이터 정제 (결측치 제거)
    df = df.dropna(subset=['Title', 'Plot', 'Release Year'])
    
    # 날짜 처리 (연도만 추출)
    try:
        df['Release Year'] = df['Release Year'].astype(str).str[-5:].str.strip()
    except:
        pass

    # 최신순 정렬 후 자르기
    try:
        df = df.sort_values(by='Release Year', ascending=False).head(LIMIT_DATA)
    except:
        df = df.head(LIMIT_DATA)
        
    df = df.reset_index(drop=True)
    print(f">> {len(df)}개의 영화 데이터 로드 완료.")
    print(f"   (가장 최신: {df['Release Year'].max()}, 가장 과거: {df['Release Year'].min()})")

    # 2. AI 모델 초기화
    engine = MovieSearchEngine()

    # 3. 임베딩 생성 (제목 + 줄거리)
    print("[Step 2] 줄거리 벡터화 시작 (잠시만 기다려주세요)...")
    
    combined_texts = (df['Title'] + ": " + df['Plot']).tolist()
    embeddings = engine.create_embeddings(combined_texts)

    # 4. 저장
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            'title': row['Title'],
            'year': row['Release Year'],
            'plot': row['Plot']
        })

    torch.save({
        'embeddings': embeddings,
        'metadata': metadata
    }, SAVE_PATH)

    print(f"[Step 3] 저장 완료! -> {SAVE_PATH}")
    print("이제 '02_main.py'를 실행하여 검색할 수 있습니다.")

if __name__ == "__main__":
    main()