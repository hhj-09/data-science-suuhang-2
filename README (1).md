# UR3 그리퍼 파지 손실 분석

제공된 UCI UR3 CobotOps 데이터를 읽어 센서 탐색과 두 분류 모델의 평가 결과를 보여 주는 Streamlit 앱입니다. Python 3.12에서 실행합니다.

## 들어 있는 파일

| 경로 | 역할 |
|---|---|
| `main.py` | 실행 시작점, 앱 제목·아이콘, 사이드바 이동 |
| `pages/1_탐색.py` | 원본, 24개 컬럼 설명, 결측값·이상치 처리, 분포, 상관계수 |
| `pages/2_분류_모델.py` | 두 모델 학습과 평가, 혼동 행렬, 결과 다운로드 |
| `analysis.py` | 전처리와 시간순 작업 주기 분할, 공통 계산 |
| `ui.py` | 데이터 캐시와 그래프 표시 |
| `requirements.txt` | 실행에 필요한 Python 라이브러리 |
| `.streamlit/config.toml` | 앱 색상과 글꼴 설정 |
| `data/ur3_cobotops.csv` | 이번에 첨부한 원본 CSV |
| `docs/수행평가_보고서.docx` | 12개 항목의 편집 가능한 보고서 |
| `docs/실제_요청_프롬프트.txt` | 이번 구현을 요청한 사용자 프롬프트 원문 |

## GitHub에 올리기

1. ZIP 파일을 압축 해제합니다. `main.py`와 `requirements.txt`가 보이는 위치가 프로젝트의 최상위 폴더입니다.
2. GitHub에 로그인한 뒤 새 저장소를 만듭니다. 저장소 이름은 직접 정합니다.
3. 저장소의 **Add file → Upload files**에서 프로젝트의 파일과 폴더를 올리고 **Commit changes**를 누릅니다. 새 빈 저장소에서는 **uploading an existing file** 링크를 사용할 수 있습니다.
4. 업로드 후 저장소 최상위에 `main.py`, `analysis.py`, `ui.py`, `requirements.txt`가 있는지 확인합니다. `pages`와 `data` 안의 파일도 확인합니다.
5. 숨김 폴더가 빠졌다면 GitHub의 **Create new file**에서 `.streamlit/config.toml`을 만들고 압축에 포함된 설정 내용을 붙여 넣습니다. `docs`는 보고서 작성에 쓰는 자료이며 앱 구동에는 코드와 데이터가 필요합니다.

## Streamlit Community Cloud에서 배포하기

1. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속해 GitHub 계정을 연결합니다.
2. **Create app**을 누르고 기존 저장소에서 배포하는 선택지인 **Yup, I have an app**을 선택합니다.
3. Repository에 업로드한 저장소, Branch에 실제 사용한 브랜치를 선택합니다. 일반적으로 브랜치는 `main`입니다.
4. Main file path에 **`main.py`**를 입력합니다. Advanced settings에서 Python **3.12**를 선택합니다.
5. Deploy를 누르고 설치·실행이 끝날 때까지 기다립니다. 앱이 열리면 주소창의 실제 `https://…streamlit.app` 주소를 복사합니다.
6. 프로젝트, 탐색, 분류 모델 페이지를 각각 확인합니다. 별도 브라우저나 시크릿 창에서도 접근되는지 확인한 뒤 보고서 10번에 실제 주소를 붙여 넣습니다. 열람 권한이 필요한 상태라면 제출 대상이 접근할 수 있게 설정합니다.

공개 배포는 아직 수행하지 않았습니다. 보고서의 주소란은 실제 배포 후 작성합니다.

[Streamlit 공식 배포 안내](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## 컴퓨터에서 먼저 실행하기

압축을 푼 폴더에서 터미널을 열고 아래 명령을 차례대로 실행합니다.

```sh
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

터미널에 표시된 Local URL을 엽니다. 서버를 끝낼 때는 터미널에서 `Ctrl+C`를 누릅니다.

- `ModuleNotFoundError`가 나타나면 같은 Python 환경에서 requirements 설치가 완료되었는지 확인합니다.
- 파일을 찾지 못하면 `data/ur3_cobotops.csv`와 파일명 대소문자를 확인합니다.
- 배포 중 오류가 나면 로그의 첫 오류와 Python 버전을 확인합니다.

## 분석 방법

원본 7,409행 중 센서 19개가 모두 비어 있는 46행과 18개가 비어 있는 8행을 제외해 7,355행을 사용합니다. 제외된 54행은 모두 정상입니다. 첨부된 `dataset_02052023.xlsx - data.csv`를 바이트 변경 없이 `data/ur3_cobotops.csv`로 복사했습니다. 실행 중 다른 데이터는 읽지 않습니다. 분석 중에는 `Temperature_T0`를 `Temperature_J0`로 대응시키고 `cycle `의 끝 공백을 제거합니다.

작업 주기를 첫 관측 시각으로 정렬해 앞 192개 주기 5,468행으로 학습하고 뒤 48개 주기 1,887행에서 평가합니다. 두 구간은 주기가 중복되지 않고 시간 경계가 겹치지 않는지 코드에서 확인합니다. 학습 통계는 학습 자료에서만 구합니다. 이 설정은 평가 결과를 보며 조정하지 않았습니다.

극단값의 영향을 줄이기 위해 두 모델의 Pipeline 첫 단계에 `QuantileClipper`를 넣었습니다. 각 센서의 학습 자료에서 0.5%·99.5% 분위수를 계산하고, 이 범위 밖 값은 가까운 경계값으로 바꿉니다. 이 비율은 평가 전에 고정했습니다. 학습에서는 575행의 센서값 870개, 평가에서는 1,797행의 센서값 7,217개가 바뀌었습니다. 이 처리로 행이나 정답을 삭제하지 않았습니다.

이 값들은 **통계적 극단값 후보**이며 측정 오류로 확인된 것은 아닙니다. 후반부 온도가 학습 구간보다 높아 평가 구간의 변경량이 많습니다. 예를 들어 Temperature_J2는 평가 1,887행 중 1,778행이 경계를 넘었습니다. 평가 자료에서 경계를 다시 계산하지 않았기 때문에 변경 비율은 1%를 넘을 수 있습니다. 실제 동작 신호가 제한될 수 있는 이 방법의 한계를 고려해야 합니다. 앱의 이상치 처리 탭에서 센서별 경계·변경량·원래 값과 처리값을 확인할 수 있습니다.

로지스틱 회귀는 표준화와 `class_weight="balanced"`를 사용합니다. 의사결정트리는 동일한 클래스 가중치, 깊이 5, 최소 말단 표본 10, 난수 시드 42를 사용합니다. 평가 자료를 바탕으로 설정을 반복 조정하려면 별도의 검증 구간이 필요합니다.

| 모델 | 정확도 | 정밀도 | 재현율 | F1 |
|---|---:|---:|---:|---:|
| 로지스틱 회귀(확률로 답하는 모델) | 0.7138 | 0.0534 | 0.5273 | 0.0970 |
| 의사결정트리(질문으로 답하는 모델) | 0.7356 | 0.0394 | 0.3455 | 0.0708 |

양성은 손실입니다. 평가 구간의 정상은 1,832행, 손실은 55행입니다. 모든 행을 정상으로 판별하는 기준 모델의 정확도는 0.9709, 손실 재현율은 0입니다. 관측 행은 실제 사고 건수와 구별해 해석해야 합니다. 한 사건이 연속된 여러 행에 기록될 수 있습니다.

센서값이 기록된 시점의 상태를 분류합니다. 미래 손실의 사전 경보 성능이나 다른 로봇 환경에서의 성능은 추가 검증이 필요합니다. 센서 상관관계의 인과적 해석 역시 별도 검증이 필요합니다.

## 보고서와 진행 일지 작성

보고서는 첨부 양식의 12개 항목 순서로 구성했습니다. 2번, 7번의 독립·종속변수 선정 이유, 8번의 해석·결론, 9번, 11번, 12번에는 학생 직접 작성 공간이 있습니다. 실제 경험과 분석에 대한 판단을 작성합니다.

진행 일지 1번에는 해당 차시에 본인이 실제 수행한 일을 기록합니다. 데이터 확인, 그래프 비교, 모델 실행, 코드 수정, 배포 중 수행한 항목을 구체적으로 적습니다.

진행 일지 2번에는 `docs/실제_요청_프롬프트.txt`의 실제 요청 원문을 복사해 사용할 수 있습니다. 목적란에는 이 요청의 목적을 적습니다. 해당 차시에 추가로 입력한 수정 프롬프트가 있다면 원문을 함께 기록합니다. 실제 입력한 차시에 맞춰 기록하세요.

진행 일지 3번에는 완성된 앱에서 직접 캡처한 화면을 넣습니다. 아래 장면이 과정을 확인하기 좋습니다.

| 캡처 위치 | 화면에 포함할 내용 |
|---|---|
| 프로젝트 | 앱 제목과 세 개의 데이터 요약 수치 |
| 탐색 → 결측값 | 처리 전후 행 수, 처리 이유, 결측값 표 |
| 탐색 → 이상치 처리 | 학습·평가별 변경량, 센서 경계값, 처리 전후 상자그림 |
| 탐색 → 상태·분포 | 정상·손실 비율, 선택한 센서 이름, 히스토그램 또는 상자그림 |
| 탐색 → 변수 간 관련성 | 선택 센서 이름과 상관계수 그래프 |
| 분류 모델 | 학습·평가 구간 표, 두 모델의 평가 지표 |
| 분류 모델 하단 | 모델 이름과 실제·예측 상태가 보이는 두 혼동 행렬 |

Windows의 `Win+Shift+S`로 필요한 영역을 캡처하고 일지에 이미지로 붙여 넣습니다. 주소 링크만 넣는 경우와 구분해 화면 이미지를 넣고, 지난 차시와 같은 화면의 중복을 피합니다.

## 실행 확인 범위

Python 3.12와 requirements의 버전으로 모델 학습을 실행했습니다. Streamlit AppTest로 홈·탐색·분류 모델 페이지 이동, 원본 행 수 선택, 센서 선택, 처리 전후 비교, 히스토그램·상자그림 전환을 확인했습니다. 극단값 경계와 표준화가 학습 자료에서만 계산되는지도 검사했습니다. 평가 행 수와 혼동 행렬 합계도 검사했습니다. 작업 환경에서 Chromium 실행이 제한되어 브라우저 화면의 시각 검증과 공개 배포 후 접속 확인은 완료하지 못했습니다.

## 데이터 및 구현 출처

Tyrovolas, M., Aliev, K., Antonelli, D., & Stylios, C. (2024). *UR3 CobotOps* [Dataset]. UCI Machine Learning Repository. 제공된 원자료를 앱에 포함했으며 분석 결과와 보고서는 이 파일에서 계산했습니다. 데이터는 CC BY 4.0으로 제공됩니다. [데이터](https://doi.org/10.24432/C5J891)

Tyrovolas, M., Stylios, C., Aliev, K., & Antonelli, D. (2024). *Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable Fault Diagnosis in Industrial Robotics*. 전류 A, 온도 °C, 속도 °/s와 상태 정의는 표 1을 참고했습니다. 첨부 CSV에 단위 메타데이터가 없어 속도값은 변환하지 않았습니다. [소개 논문](https://doi.org/10.1007/978-3-031-63851-0_6)

[로지스틱 회귀 공식 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html) · [의사결정트리 공식 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[이상치 해석 · NIST](https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) · [윈저라이징 개념 · SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html) · [전처리와 데이터 누출 · scikit-learn](https://scikit-learn.org/1.8/common_pitfalls.html)
