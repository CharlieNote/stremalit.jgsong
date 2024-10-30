import streamlit as st
import pandas as pd
from sentence_transformers import SenrtenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "종관이형 사는 곳 어디인가요?",
    "종관이형의 차는 무엇인가요?",
    "종관이형의 현재 어디 팀 인가요?",
    "종관이형의 나이는 몇살인가요?",
    "종관이형은 여자친구가 있나요?"
]

answers = [
    "서울시 서초구 양재동 입니다.",
    "크루즈 송 입니다. 여자아니면 태우질 않습니다.",
    "광고사업부 입니다.",
    "만 28세 일 겁니다",
    "없는데 최근에 읍읍읍!!"
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("이력서 챗봇")
st.write("종게형에게 무엇이든 물어보세요.(5질문) 사는곳, 차, 어디팀, 나이, 여자친구 유무)

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
