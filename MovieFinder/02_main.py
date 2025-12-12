import sys
from modules.embedder import MovieSearchEngine
from modules.ui import show_result_card

INDEX_PATH = 'movie_index.pt'

def main():
    print("=========================================")
    print("            🎬 MovieFinder              ")
    print("=========================================")

    # 1. 엔진 초기화
    engine = MovieSearchEngine()
    
    if not engine.load_index(INDEX_PATH):
        print(f"[Error] '{INDEX_PATH}' 파일이 없습니다.")
        print("'01_build_index.py'를 먼저 실행해주세요.")
        return

    # 2. 검색 루프
    while True:
        print("\n" + "-"*40)
        query = input("🔍 묘사할 장면을 영어로 입력하세요 (종료: q)\n>> ")

        if query.lower() == 'q':
            print("MovieFinder를 종료합니다.")
            break
        
        if len(query) < 3:
            print("입력이 너무 짧습니다.")
            continue

        # 3. 검색 수행
        print("   Searching...")
        results = engine.search(query, top_k=1)
        best_match = results[0]

        # 4. 결과 출력
        print(f"\n[Result] {best_match['title']} ({best_match['year']})")
        print(f"[Score]  {best_match['score']*100:.2f}%")
        print(f"[Plot]   {best_match['plot'][:100]}...")

        # 5. OpenCV 시각화
        print(">> 결과 창이 떴습니다. (닫으려면 아무 키나 누르세요)")
        show_result_card(best_match)

if __name__ == "__main__":
    main()