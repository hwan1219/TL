import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# .\venv_tf\Scripts\Activate

# 1. 모델 불러오기
# 최초 1회만 실행되게 캐싱
@st.cache_resource # 데코레이터 - 기존 함수를 수정하지 않고 기능을 확장
def load_trained_model():
  model = load_model('cat_vs_dog_model.h5', compile=False)
  return model
model = load_trained_model()

# 2. 페이지 제목
st.title("🐶 vs 🐱 판별기")

# 3. 파일 업로드
uploaded_file = st.file_uploader("강아지 또는 고양이 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  # 4. 이미지 보여주기
  img = Image.open(uploaded_file)
  st.image(img, use_container_width=True)

  # 5. 분석 버튼
  if st.button("분석하기"):
  # javascript 같은 언어와는 다르게 내부적으로 버튼 생성과 클릭 감지 로직이 다 짜여있음
  
    # 6. 이미지 전처리
    img = img.resize((150, 150)) # 전이학습 모델 크기에 맞춤
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # 예측을 위한 샘플 수 차원 추가
    img_array = img_array / 255.0 # 정규화

    # 7. 예측
    prediction = model.predict(img_array)[0][0]
    # 이진 분류 확률, ex) [[0.2]] - 샘플 수 차원으로 인한 2차원 배열 반환
    
    dog_prob = prediction # 1 = 강아지
    cat_prob = 1 - prediction # 0 = 고양이

    # 8. 결과 출력
    st.subheader("📊 예측 결과")
    st.write(f"🐶 강아지일 확률: {dog_prob * 100:.2f}%")
    st.write(f"🐱 고양이일 확률: {cat_prob * 100:.2f}%")

    label = "강아지" if dog_prob > cat_prob else "고양이"
    st.success(f"이 이미지는 **{label}**입니다!")