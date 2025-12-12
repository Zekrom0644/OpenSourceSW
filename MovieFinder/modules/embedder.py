import torch
from sentence_transformers import SentenceTransformer, util

class MovieSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Hugging Face SBERT 모델 초기화
        """
        print(f"[Module] AI 모델 로딩 중... ({model_name})")
        # CPU/GPU 자동 감지
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings = None
        self.metadata = None

    def create_embeddings(self, texts):
        """
        텍스트 리스트를 벡터로 변환
        """
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    def load_index(self, filepath):
        """
        저장된 임베딩 데이터(.pt) 불러오기
        """
        try:
            data = torch.load(filepath, map_location=self.device)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata'] 
            print(f"[Module] 인덱스 로드 완료: {len(self.metadata)}개 영화")
            return True
        except FileNotFoundError:
            return False

    def search(self, query, top_k=1):
        """
        질문과 가장 유사한 영화 검색
        """
        if self.embeddings is None:
            raise Exception("인덱스가 로드되지 않았습니다.")

        # 1. 쿼리 벡터화
        query_vec = self.model.encode(query, convert_to_tensor=True)

        # 2. 코사인 유사도 계산
        cos_scores = util.cos_sim(query_vec, self.embeddings)[0]

        # 3. 상위 k개 추출
        top_results = torch.topk(cos_scores, k=top_k)
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            idx = int(idx)
            results.append({
                'title': self.metadata[idx]['title'],
                'year': self.metadata[idx]['year'],
                'plot': self.metadata[idx]['plot'],
                'score': float(score)
            })
        return results