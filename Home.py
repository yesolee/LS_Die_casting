import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="🔥",
    layout="wide"
)

with st.sidebar:
    st.header('목차')
    st.markdown('1️⃣ 개요')
    st.markdown('2️⃣ EDA')    
    st.markdown('3️⃣ 전처리')
    st.markdown('4️⃣ 모델비교')    
    st.markdown('5️⃣ 결론')    

st.write("# 🔥다이캐스팅 공정 불량 판정 모델 개선")
with st.container():
    col1, col2 = st.columns([1,1])
    col1.image(Image.open('img/die_casting.jpg'))

    col2.subheader('🚩 분석배경')
    col2.success('- **2019년 02월 15일 불량률을 낮추는 새로운 시스템 도입** \n - **시스템 도입 후 5주간 모니터링 결과, 불량률 감소 확인** ')
    col2.subheader('🤔 문제정의')
    col2.error('- **기존 불량 예측 모델에 대한 의심** \n - **기존의 운영 관리 방향에 대한 점검 필요**')
    col2.subheader('🙆‍♂️ 기대효과')
    col2.info('- **새로운 운영 관리 방향 수립** \n - **공장 효율 증가, 비용 절감** ')


with st.container():
    st.subheader('📝 분석방법')
    st.write('#### 1️⃣ PSI 분포 확인 (데이터의 분포 차이를 검증)')
    col1, col2 = st.columns([1,1])
    col1.image(Image.open('img/psi.jpg'))
    col2.warning('- **분석 시작점: 2019.03.25** \n - **ActualProp: 2019.01.02.~ 2019.02.14 (시스템 도입 전)** \n - **ExpectedProp: 2019.02.15 ~ 2019.03.24**')
    st.write('#### 2️⃣ 모델 생성 및 비교 (기존 모델, 신규 데이터, 기존+신규 데이터 )')
    st.image(Image.open('img/models.jpg'))



    