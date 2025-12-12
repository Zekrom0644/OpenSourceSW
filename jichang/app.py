import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="Tube-Insight", page_icon="🎬")

st.title("🎬 Tube-Insight: 유튜브 댓글 감성 분석기")
st.markdown("유튜브 링크를 넣으면 **AI가 댓글 반응(긍정/부정)**을 분석해줍니다.")

# 2. AI 모델 로드 (캐싱을 사용하여 속도 향상)
@st.cache_resource
def load_model():
    # 한국어 감성 분석에 특화된 모델 다운로드 (최초 1회 시간 소요)
    # 모델: matthewburke/korean_sentiment (SKT KoBERT 기반 등)
    return pipeline("text-classification", model="matthewburke/korean_sentiment")

with st.spinner("AI 모델을 불러오는 중입니다... (최초 실행 시 1~2분 소요)"):
    classifier = load_model()

# 3. 사용자 입력
url = st.text_input("분석할 유튜브 영상 링크(URL)를 입력하세요:")

if url:
    try:
        downloader = YoutubeCommentDownloader()
        comments = []
        limit = 50  # 시간 절약을 위해 50개만 분석 (필요시 수정)
        
        with st.spinner(f"최근 댓글 {limit}개를 수집하고 분석 중입니다..."):
            # 댓글 수집 및 분석
            generator = downloader.get_comments_from_url(url, sort_by=1) # 최신순
            
            count = 0
            for comment in generator:
                text = comment['text']
                if not text: continue
                
                # AI 분석 수행
                result = classifier(text)[0] # {'label': 'LABEL_1', 'score': 0.9}
                
                # 라벨 변환 (모델마다 다름, 이 모델은 1:긍정, 0:부정)
                label = "긍정 😊" if result['label'] == 'LABEL_1' else "부정 😠"
                score = round(result['score'] * 100, 2)
                
                comments.append([text, label, score])
                
                count += 1
                if count >= limit:
                    break
        
        # 4. 결과 시각화
        if comments:
            df = pd.DataFrame(comments, columns=['댓글 내용', '감성', '확신도(%)'])
            
            # 통계 보여주기
            col1, col2 = st.columns(2)
            pos_count = len(df[df['감성'] == "긍정 😊"])
            neg_count = len(df[df['감성'] == "부정 😠"])
            
            col1.metric("긍정 댓글", f"{pos_count}개")
            col2.metric("부정 댓글", f"{neg_count}개")
            
            # 차트 그리기
            st.bar_chart(df['감성'].value_counts())
            
            # 데이터 표 보여주기
            st.subheader("상세 분석 결과")
            st.dataframe(df)
            
        else:
            st.warning("댓글을 가져올 수 없습니다. 링크를 확인해주세요.")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.info("Tip: 'https://www.youtube.com/watch?v=...' 형식의 링크인지 확인하세요.")