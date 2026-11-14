
"""페이지에서 함께 사용하는 데이터 로딩과 그래프 표시 설정."""

import streamlit as st

from analysis import (
    load_raw,
    prepare_data,
    complete_rows,
    evaluate,
    outlier_processing
)


# ==================================================
# 1. 그래프 색상 설정
# ==================================================

COLORS = {
    "정상": "#157F83",
    "파지 손실": "#DD6B3E"
}


# ==================================================
# 2. 데이터 불러오기
# ==================================================

@st.cache_data(show_spinner=False)
def get_data():

    # 원본 CSV 불러오기
    raw = load_raw()

    # 데이터 전처리
    prepared = prepare_data(raw)

    # 결측값이 있는 행 제거
    clean = complete_rows(prepared)

    return raw, prepared, clean


# ==================================================
# 3. 분류 모델 학습 및 평가
# ==================================================

@st.cache_data(
    show_spinner="로지스틱 회귀와 의사결정트리를 학습하고 있습니다."
)
def get_evaluation(data):

    """
    analysis.py의 evaluate 함수를 실행한다.

    반환값:
    1. scores         : 모델별 평가 지표
    2. matrices       : 모델별 혼동 행렬
    3. predictions    : 평가 데이터별 예측 결과
    4. fitted_models  : 학습이 완료된 모델
    """

    result = evaluate(data)

    return result


# ==================================================
# 4. 극단값 처리 데이터
# ==================================================

@st.cache_data(show_spinner=False)
def get_outlier_data(data):

    """
    극단값 처리 결과를 가져온다.

    반환값:
    1. processed : 극단값 처리 데이터
    2. bounds    : 센서별 극단값 경계
    3. audit     : 처리 결과 요약
    4. changed   : 변경된 센서값 여부
    """

    return outlier_processing(data)


# ==================================================
# 5. 그래프 공통 설정
# ==================================================

def chart(figure, key=None):

    """
    Plotly 그래프의 공통 디자인을 적용한다.
    """

    figure.update_layout(

        font={
            "family": "sans-serif",
            "size": 13
        },

        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        },

        legend={
            "orientation": "h",
            "y": -0.2,
            "title_text": ""
        },

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)"

    )

    st.plotly_chart(

        figure,

        width="stretch",

        key=key,

        config={
            "displaylogo": False,
            "scrollZoom": False
        }

    )


# ==================================================
# 6. CSV 다운로드 기능
# ==================================================

def csv_bytes(table):

    """
    데이터프레임을 CSV 파일로 변환한다.

    utf-8-sig를 사용하여 한글이 깨지지 않도록 한다.
    """

    return table.to_csv(
        index=False
    ).encode("utf-8-sig")
