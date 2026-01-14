import streamlit as st
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_model(model_name):
    return genai.GenerativeModel(model_name)

def classify_image(prompt, image, model_name):
    model = get_model(model_name)
    response = model.generate_content([prompt, image])
    return response.text

st.set_page_config(
    page_title="Image Classification- Gemini",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('이미지 분류기- Gemini')

with st.sidebar:
    model = st.selectbox(
        "모델 선택",
        options=['gemini-2.0-flash'],
        index=0
    )

prompt = """
이미지를 보고 다음 보기 내용이 포함되면 1, 포함되지 않으면 0으로 분류해줘.
보기 = [건축물, 바다, 산]
JSON format으로 키는 'building', 'sea', 'mountain'으로 하고 각각 건축물, 바다, 산에 대응되도록 출력해줘.
자연 이외의 건축물이 조금이라도 존재하면 'building'을 1로,
물이 조금이라도 존재하면 'sea'를 1로,
산이 조금이라도 보이면 'mountain'을 1로 설정해줘.
markdown format은 포함하지 말아줘.
"""

uploaded_file = st.file_uploader('이미지 업로드', type=['jpg','jpeg','png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='업로드한 이미지', use_container_width=True)

    if st.button('분류 실행'):
        with st.spinner('분류 중...'):
            response = classify_image(prompt, img, model)

        st.subheader('분류 결과')
        st.code(response)
