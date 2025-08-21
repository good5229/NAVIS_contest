# BDS (Balanced Development Score) - 지역 균형발전 지수

## 프로젝트 개요

이 프로젝트는 한국의 17개 시도별 균형발전 수준을 측정하는 BDS(Balanced Development Score) 지수를 개발하고 분석하는 시스템입니다. KOSIS(통계청) API를 활용하여 실시간 데이터를 수집하고, 다양한 가중치 방식을 통해 지역발전의 균형성을 종합적으로 평가합니다.

## 주요 기능

### 1. BDS 지수 계산
- **4개 핵심 지표**를 활용한 종합 평가:
  - 1인당 지역내총생산 (경제력)
  - 인구증가율 (성장성)
  - 고령인구비율 (사회구조)
  - 재정자립도 (재정건전성)

### 2. 가중치 방식
- **PCA 가중치**: 주성분분석을 통한 데이터 기반 가중치
- **동일 가중치**: 모든 지표에 동일한 가중치 적용
- **사용자 정의 가중치**: 사용자가 직접 가중치 설정

### 3. 시각화
- **시계열 분석**: 최근 10년간의 BDS 변화 추이
- **지도 시각화**: Plotly 기반 인터랙티브 코로플레스 지도
- **애니메이션**: 연도별 변화를 애니메이션으로 표현

## 설치 및 설정

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv navis
source navis/bin/activate  # Windows: navis\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정
`.env` 파일을 생성하고 KOSIS API 키를 설정하세요:
```
KOSIS_API_KEY=your_api_key_here
```

## 사용법

### 기본 실행
```python
from bds_analysis import build_timeseries_and_map

# PCA 가중치로 BDS 계산 및 시각화
build_timeseries_and_map(
    weight_mode="pca",
    min_year=2000,
    geojson_path="./skorea-provinces-2018-geo.json"
)
```

### 가중치 방식 변경
```python
# 동일 가중치
build_timeseries_and_map(weight_mode="equal")

# 사용자 정의 가중치
custom_weights = {
    "pc_grdp": 0.35,
    "pop_growth": 0.25,
    "elderly_rate": 0.15,
    "fiscal_indep": 0.25
}
build_timeseries_and_map(weight_mode="custom", custom_weights=custom_weights)
```

## 출력 파일

실행 후 `outputs_timeseries/` 폴더에 다음 파일들이 생성됩니다:

- `bds_timeseries.csv`: 시계열 BDS 데이터
- `bds_choropleth.html`: 인터랙티브 지도 시각화
- `weights_by_year.json`: 연도별 가중치 정보

## 데이터 소스

- **KOSIS API**: 통계청 공식 통계 데이터
- **지역발전지수**: NAVIS에서 제공하는 지역발전지수 데이터
- **지도 데이터**: 한국 시도 경계 GeoJSON

## 분석 결과 활용

1. **정책 수립**: 지역별 균형발전 정책의 우선순위 설정
2. **투자 의사결정**: 지역별 투자 효과성 분석
3. **연구 활용**: 지역발전 관련 학술 연구 및 논문 작성

## 기술 스택

- **Python 3.12+**
- **pandas**: 데이터 처리 및 분석
- **scikit-learn**: PCA 및 표준화
- **plotly**: 인터랙티브 시각화
- **requests**: API 호출
- **numpy**: 수치 계산

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 문의사항

프로젝트에 대한 문의사항이나 개선 제안이 있으시면 이슈를 등록해 주세요.