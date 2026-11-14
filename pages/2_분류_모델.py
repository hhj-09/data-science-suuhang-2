
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    LR, TREE, TARGET, FEATURES, LABELS,
    split_cycles, split_table
)

from ui import (
    chart, csv_bytes, get_data, get_evaluation
)

st.title("분류 모델과 평가")

# --------------------------------------------------
# 1. 데이터 불러오기
# --------------------------------------------------

_, _, data = get_data()

train, test = split_cycles(data)

st.write(
    "UR3 로봇팔의 19개 센서 데이터를 활용하여 "
    "그리퍼의 정상 상태와 파지 손실 상태를 분류합니다."
)

st.info(
    "정상(False=0), 파지 손실(True=1)을 구분하는 "
    "이진 분류 문제입니다."
)

st.markdown("### 학습에 사용하는 센서")

st.write(FEATURES)

# --------------------------------------------------
# 2. 모델 학습 방식
# --------------------------------------------------

with st.expander("두 모델의 학습 방식", expanded=True):

    st.markdown("#### 로지스틱 회귀")

    st.write(
        "센서값을 표준화하고 각 센서의 가중치를 계산합니다. "
        "이를 바탕으로 파지 손실일 확률을 계산합니다. "
        "확률이 0.5 이상이면 파지 손실, "
        "0.5 미만이면 정상으로 분류합니다."
    )

    st.markdown("#### 의사결정트리")

    st.write(
        "센서값을 기준으로 여러 질문을 만들어 데이터를 나눕니다. "
        "각 질문을 따라가 최종적으로 정상 또는 파지 손실을 분류합니다."
    )

    st.write(
        "두 모델 모두 동일한 학습 데이터와 평가 데이터를 사용합니다."
    )

# --------------------------------------------------
# 3. 학습 데이터와 평가 데이터 분리
# --------------------------------------------------

st.markdown("### 학습·평가 데이터 분할")

st.dataframe(
    split_table(train, test),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "작업 주기의 시간순으로 앞 80%를 학습에 사용하고 "
    "뒤 20%를 평가에 사용합니다."
)

# --------------------------------------------------
# 4. 실제 모델 학습 및 예측
# --------------------------------------------------

scores, matrices, predictions, fitted_models = (
    get_evaluation(data)
)

st.markdown("### 두 모델의 실제 평가 결과")

st.dataframe(
    scores.style.format("{:.4f}"),
    use_container_width=True
)

st.download_button(
    "평가 지표 CSV 다운로드",
    csv_bytes(scores.reset_index()),
    "model_metrics.csv",
    "text/csv"
)

# --------------------------------------------------
# 5. 모델별 평가 지표
# --------------------------------------------------

st.markdown("### 평가 지표 비교")

metric_names = [
    "정확도(Accuracy)",
    "정밀도(Precision)",
    "재현율(Recall)",
    "F1 점수"
]

for metric in metric_names:

    st.markdown(f"#### {metric}")

    fig = px.bar(
        scores.reset_index(),
        x="모델",
        y=metric,
        text_auto=".4f",
        title=f"모델별 {metric} 비교"
    )

    fig.update_yaxes(range=[0, 1])

    chart(fig, key=metric)

# --------------------------------------------------
# 6. 혼동 행렬
# --------------------------------------------------

st.markdown("### 실제 정답과 예측 결과 비교")

for index, (name, matrix) in enumerate(matrices.items()):

    st.markdown(f"#### {name}")

    matrix_df = pd.DataFrame(
        matrix,
        index=["실제 정상", "실제 파지 손실"],
        columns=["예측 정상", "예측 파지 손실"]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    fig = px.imshow(
        matrix,
        x=["예측 정상", "예측 파지 손실"],
        y=["실제 정상", "실제 파지 손실"],
        text_auto=True,
        color_continuous_scale="Teal",
        labels={
            "x": "모델 예측",
            "y": "실제 정답",
            "color": "행 수"
        },
        title=f"{name} 혼동 행렬"
    )

    chart(fig, key=f"matrix_{index}")

    st.write(
        f"실제 파지 손실을 찾아낸 행: {matrix[1, 1]:,}개"
    )

    st.write(
        f"파지 손실을 놓친 행: {matrix[1, 0]:,}개"
    )

    st.write(
        f"정상을 손실로 잘못 분류한 행: {matrix[0, 1]:,}개"
    )

# --------------------------------------------------
# 7. 실제 정답과 예측 결과 비교
# --------------------------------------------------

st.markdown("### 모델이 실제로 분류한 결과")

comparison = pd.DataFrame({
    "구분": [
        "실제 정상",
        "실제 파지 손실",
        "로지스틱 회귀 예측 정상",
        "로지스틱 회귀 예측 파지 손실",
        "의사결정트리 예측 정상",
        "의사결정트리 예측 파지 손실"
    ],
    "행 수": [
        int((test[TARGET] == False).sum()),
        int((test[TARGET] == True).sum()),
        int((predictions[LR] == False).sum()),
        int((predictions[LR] == True).sum()),
        int((predictions[TREE] == False).sum()),
        int((predictions[TREE] == True).sum())
    ]
})

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True
)

fig = px.bar(
    comparison,
    x="구분",
    y="행 수",
    text_auto=True,
    title="실제 정답과 두 모델의 예측 개수 비교"
)

chart(fig, key="prediction_comparison")

# --------------------------------------------------
# 8. 개별 데이터 분류 결과
# --------------------------------------------------

st.markdown("### 평가 데이터별 실제 분류 결과")

result = predictions.copy()

# 실제 정답을 한글로 변환
result["실제 상태"] = result[TARGET].map(LABELS)

# 각 모델의 예측 상태를 한글로 변환
result["로지스틱 회귀 예측"] = (
    result[LR].map(LABELS)
)

result["의사결정트리 예측"] = (
    result[TREE].map(LABELS)
)

result["로지스틱 회귀 정답 여부"] = (
    result[f"{LR} 정답 여부"]
)

result["의사결정트리 정답 여부"] = (
    result[f"{TREE} 정답 여부"]
)

display_columns = [
    "Num",
    "Timestamp",
    "cycle",
    "실제 상태",
    "로지스틱 회귀 예측",
    f"{LR} 손실 확률",
    "로지스틱 회귀 정답 여부",
    "의사결정트리 예측",
    f"{TREE} 손실 확률",
    "의사결정트리 정답 여부"
]

st.dataframe(
    result[display_columns],
    hide_index=True,
    use_container_width=True,
    height=450
)

st.download_button(
    "전체 분류 결과 CSV 다운로드",
    csv_bytes(result[display_columns]),
    "classification_results.csv",
    "text/csv"
)

# --------------------------------------------------
# 9. 로지스틱 회귀의 센서별 가중치
# --------------------------------------------------

st.markdown("### 로지스틱 회귀 분석")

lr_model = fitted_models[LR].named_steps[
    "logisticregression"
]

coef_df = pd.DataFrame({
    "센서": FEATURES,
    "가중치": lr_model.coef_[0]
})

coef_df["절댓값"] = coef_df["가중치"].abs()

coef_df = coef_df.sort_values(
    "절댓값",
    ascending=False
)

st.write(
    "가중치의 부호와 크기를 통해 "
    "각 센서가 파지 손실 분류에 어떻게 반영되었는지 확인할 수 있습니다."
)

st.dataframe(
    coef_df.drop(columns="절댓값"),
    hide_index=True,
    use_container_width=True
)

fig = px.bar(
    coef_df,
    x="가중치",
    y="센서",
    orientation="h",
    title="로지스틱 회귀 센서별 가중치"
)

chart(fig, key="logistic_coefficients")

# --------------------------------------------------
# 10. 의사결정트리의 실제 질문 규칙
# --------------------------------------------------

st.markdown("### 의사결정트리 분석")

from sklearn.tree import export_text

tree_model = fitted_models[TREE].named_steps[
    "decisiontreeclassifier"
]

rules = export_text(
    tree_model,
    feature_names=FEATURES,
    decimals=3
)

st.write(
    "아래는 학습된 의사결정트리가 실제로 사용하는 질문 규칙입니다."
)

st.code(rules, language="text")

importance_df = pd.DataFrame({
    "센서": FEATURES,
    "중요도": tree_model.feature_importances_
}).sort_values(
    "중요도",
    ascending=False
)

st.dataframe(
    importance_df,
    hide_index=True,
    use_container_width=True
)

fig = px.bar(
    importance_df,
    x="중요도",
    y="센서",
    orientation="h",
    title="의사결정트리 센서 중요도"
)

chart(fig, key="tree_importance")

# --------------------------------------------------
# 11. 결과 해석
# --------------------------------------------------

st.markdown("### 분석 결과 해석")

st.write(
    "두 모델은 동일한 평가 데이터를 사용하여 정상과 파지 손실을 분류했습니다. "
    "정확도뿐 아니라 정밀도, 재현율, F1 점수와 혼동 행렬을 함께 비교해야 합니다."
)

st.warning(
    "이 분석은 센서가 기록된 시점의 파지 손실 상태를 분류합니다. "
    "미래의 파지 손실을 예측하는 기능과는 다릅니다. "
    "실제 로봇에 적용하기 전에는 별도의 검증이 필요합니다."
)
