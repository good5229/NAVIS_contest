# NAVIS 지역발전지수 종합 분석 대시보드

## 📋 프로젝트 개요

이 프로젝트는 NAVIS 지역발전지수와 BDS(Better Development Score)를 통합하여 지역 균형발전을 종합적으로 분석하는 웹 대시보드입니다. 한국은행 ECOS API와 KOSIS 데이터를 활용하여 실시간 경제지표를 수집하고, 재정자립도 분석과 정책 시뮬레이션을 통해 지역별 균형발전 정도를 시각화하여 정책적 시사점을 도출합니다.

## 🎯 주요 기능

### 1. 지역균형발전지수 (Regional Balanced Development Index)
- **학술적 근거 기반**: Barro & Sala-i-Martin (1992), OECD (2020) 등 기존 연구 기반 가중치 설계
- **실시간 계산**: BDS 값의 지역 간 편차를 분석하여 균형발전 정도를 점수화
- **세부 계산 항목**: 최고-최저 격차, 상위/하위 평균 차이, 표준편차, 중간값 차이
- **5단계 판정 체계**: 매우 균형, 균형, 보통, 불균형, 심한 불균형으로 세분화
- **투명한 공식**: 계산 방법을 명시적으로 표시하여 신뢰성 확보
- **연도별 트렌드**: 1997-2025년 균형발전지수 변화 추이
- **상위 30% 지역명 표시**: 해당 연도의 상위 30% 지역 이름을 직접 표시

### 2. 재정자립도 분석 (Fiscal Autonomy Analysis)
- **실시간 KOSIS 데이터**: 통계청 KOSIS API를 통한 실제 재정자립도 데이터 수집
- **지역별 현황 분석**: 17개 행정구역별 재정자립도 현황 및 격차 분석
- **연도별 트렌드**: 1997-2025년 재정자립도 변화 추이
- **격차 지수 계산**: 최고-최저 격차, 표준편차, 지니계수 등 통계적 분석
- **정책적 시사점**: 재정자립도 격차 해소를 위한 정책 제안

### 3. 정책 시뮬레이션 시스템
#### 3.1 선형 정책 시뮬레이터
- **기본 정책 효과**: 인구, 산업, 인프라, 제도 정책의 직접적 효과 모델링
- **지역별 특성 반영**: 도시/도 지역별 차별화된 투자 효과 분석
- **시각적 결과**: 투자 효과를 직관적으로 표현

#### 3.2 비선형 정책 시뮬레이터
- **지수함수 효과**: 정책 효과의 비선형적 발현 패턴 모델링
- **시차 효과**: 정책 시행부터 효과 발현까지의 시간 지연 반영
- **한계효용 체감**: 투자 증가에 따른 효과 체감 현상 반영
- **시너지 효과**: 정책 조합에 따른 상승효과 계산

#### 3.3 정책 의사결정 시뮬레이터
- **다차원 효과 분석**: 직접효과, 간접효과, 파급효과, 시너지효과 종합 분석
- **비용-효과 분석**: 직접비용, 기회비용, 관리비용을 고려한 효율성 평가
- **리스크 평가**: Monte Carlo 시뮬레이션을 통한 불확실성 분석
- **다기준 의사결정**: 효과성, 효율성, 실현가능성, 수용성, 지속가능성 평가
- **정책 우선순위**: 종합 점수 기반 정책 우선순위 결정

### 4. BDS 트렌드 분석
- **다중 지역 선택**: 체크박스를 통한 지역별 BDS 추이 비교 (4열 레이아웃)
- **예측 데이터 구분**: 2025년 예측값을 별도로 표시 (다이아몬드 마커)
- **범례 클릭 기능**: 범례에서 지역을 클릭하면 해당 지역의 모든 데이터(실제+예측) 완전히 숨김
- **Y축 범위 최적화**: 3-7 범위로 설정하여 가독성 향상

### 5. NAVIS vs BDS 상관관계 분석
- **연도별 상관관계**: 1997-2020년 NAVIS와 BDS 간 상관관계 변화
- **점진적 개선**: 상관관계가 시간이 지남에 따라 향상되는 패턴 확인
- **통계적 검증**: Pearson 상관계수를 통한 정량적 분석

### 6. 지역별 BDS 지도 시각화
- **GeoJSON 기반**: 한국 행정구역 경계를 정확히 반영
- **연도별 색상**: BDS 값에 따른 지역별 색상 구분
- **통합 호버/클릭 정보**: BDS, GDP, 인구, 경제활동인구 정보를 호버와 클릭 시 동일하게 표시
- **2025년 예측 표시**: 예측 데이터는 "(예측)" 라벨로 명확히 구분

### 7. 인터랙티브 정책 시뮬레이터
- **사용자 정의 파라미터**: 정책 강도, 기간, 투자 규모 등을 사용자가 직접 조정
- **실시간 결과 업데이트**: 파라미터 변경 시 즉시 시뮬레이션 결과 반영
- **상세 설명 모달**: 각 차트와 지표에 대한 상세한 설명 제공
- **학술적 근거 표시**: 모든 계수와 모델의 학술적 출처 명시

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv navis_env

# 가상환경 활성화
source navis_env/bin/activate  # macOS/Linux
# 또는
navis_env\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 수집
```bash
# KOSIS 재정자립도 데이터 수집
python scripts/kosis_fiscal_data_collector.py

# 재정자립도 분석 실행
python scripts/fiscal_autonomy_analyzer.py
```

### 3. 정책 시뮬레이션 실행
```bash
# 선형 정책 시뮬레이션
python simulators/fiscal_policy_simulator.py

# 비선형 정책 시뮬레이션
python simulators/nonlinear_fiscal_simulator.py

# 정책 의사결정 시뮬레이션
python simulators/policy_decision_simulator.py

# 인터랙티브 정책 시뮬레이터 생성
python simulators/interactive_policy_decision_simulator.py
```

### 4. 대시보드 실행
```bash
# 로컬 서버 실행 (CORS 이슈 해결을 위해 필요)
python3 -m http.server 8000

# 브라우저에서 접속
open http://localhost:8000/dashboards/bok_navis_comprehensive_dashboard.html
```

## 📁 파일 구조

```
NAVIS_contest/
├── dashboards/                                   # 대시보드 HTML 파일들
│   ├── bok_navis_comprehensive_dashboard.html    # 메인 종합 대시보드
│   ├── interactive_policy_decision_simulator.html # 인터랙티브 정책 시뮬레이터
│   ├── fiscal_policy_simulation_dashboard.html   # 선형 정책 시뮬레이션 결과
│   ├── linear_vs_nonlinear_comparison.html       # 선형 vs 비선형 비교
│   ├── policy_decision_dashboard.html            # 정책 의사결정 시뮬레이션 결과
│   └── interactive_fiscal_simulator.html         # 인터랙티브 재정 시뮬레이터
│
├── simulators/                                   # 정책 시뮬레이션 스크립트
│   ├── fiscal_policy_simulator.py                # 선형 정책 시뮬레이터
│   ├── nonlinear_fiscal_simulator.py             # 비선형 정책 시뮬레이터
│   ├── policy_decision_simulator.py              # 정책 의사결정 시뮬레이터
│   ├── interactive_policy_decision_simulator.py  # 인터랙티브 시뮬레이터 생성기
│   └── interactive_fiscal_simulator.py           # 인터랙티브 재정 시뮬레이터
│
├── scripts/                                      # 데이터 처리 스크립트
│   ├── kosis_fiscal_data_collector.py            # KOSIS 데이터 수집기
│   └── fiscal_autonomy_analyzer.py               # 재정자립도 분석기
│
├── data/                                         # 데이터 파일들
│   ├── fiscal_autonomy/                          # 재정자립도 관련 데이터
│   │   ├── kosis_fiscal_autonomy_data.csv        # KOSIS 재정자립도 데이터
│   │   ├── fiscal_gap_index_data.csv             # 재정 격차 지수 데이터
│   │   ├── fiscal_autonomy_data.db               # 재정자립도 데이터베이스
│   │   └── kosis_fiscal_autonomy_data.json       # KOSIS JSON 데이터
│   ├── simulation_results/                       # 시뮬레이션 결과 데이터
│   │   ├── fiscal_policy_simulation_results.csv  # 선형 시뮬레이션 결과
│   │   ├── nonlinear_fiscal_simulation_results.csv # 비선형 시뮬레이션 결과
│   │   └── policy_decision_simulation_results.csv # 정책 의사결정 시뮬레이션 결과
│   ├── navis/                                    # NAVIS 원본 데이터
│   │   ├── 1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx
│   │   └── skorea-provinces-2018-geo.json       # 한국 지도 Geojson
│   └── kosis/                                    # KOSIS 데이터
│       └── 2025년_1분기_실질_지역내총생산(잠정).xlsx
│
├── docs/                                         # 문서 파일들
│   ├── 재정자립도_정책_제안서.md                  # 재정자립도 정책 제안서
│   ├── 재정자립도_낮은_지역_원인분석_연구.md       # 재정자립도 원인 분석
│   ├── 시뮬레이션_선형성_분석_및_개선방안.md       # 시뮬레이션 개선 방안
│   ├── 정책수립_목적_시뮬레이션_개선방안.md        # 정책 수립 시뮬레이션 개선
│   └── fiscal_autonomy_analysis_report.md        # 재정자립도 분석 보고서
│
├── screenshots/                                  # 대시보드 스크린샷
├── requirements.txt                              # Python 의존성
├── .gitignore                                    # Git 제외 파일
├── LICENSE                                       # 라이선스
└── README.md                                     # 프로젝트 문서
```

## 📊 주요 분석 결과

### 재정자립도 분석 결과 (2025년 기준)
- **전국 평균**: 58.3%
- **최고 지역**: 서울특별시 (85.7%)
- **최저 지역**: 충청남도 (28.1%)
- **최고-최저 격차**: 57.6%p
- **높은 재정자립도 지역 (70% 이상)**: 서울, 경기, 세종, 인천
- **낮은 재정자립도 지역 (50% 미만)**: 충청남도, 경상남도, 강원도, 충청북도, 경상북도, 제주도

### 지역균형발전지수 계산 공식 (학술적 근거 기반)
```
균형발전 지수 = 격차 점수 × 0.35 + 균형 점수 × 0.30 + 분산 점수 × 0.20 + 중간값 점수 × 0.15

개별 점수 계산:
- 격차 점수: max(0, 100 - 최고-최저 격차 × 50)
- 균형 점수: max(0, 100 - 상위/하위 평균 차이 × 60)
- 분산 점수: max(0, 100 - 표준편차 × 40)
- 중간값 점수: max(0, 100 - 중간값 차이 × 50)

발전 수준 판정 (5단계):
- 81-100점: 매우 균형
- 66-80점: 균형
- 50-65점: 보통
- 25-49점: 불균형
- 0-24점: 심한 불균형
```

### 정책 시뮬레이션 모델
#### 선형 모델
```
재정자립도 개선 = Σ(정책 효과 계수 × 정책 강도 × 지역 특성 계수)
```

#### 비선형 모델
```
재정자립도 개선 = Σ(기본 효과 × (1 - e^(-감쇠율 × 시간)) × (1 - 한계효용 체감률 × 누적 투자))
```

#### 정책 의사결정 모델
```
종합 점수 = 효과성(0.3) + 효율성(0.25) + 실현가능성(0.2) + 수용성(0.15) + 지속가능성(0.1)
```

### BDS 모델 특징
- **데이터 소스**: ECOS + KOSIS (NAVIS 제외)
- **분석 기간**: 1997-2025년 (29년간)
- **예측 기간**: 2025년 (ECOS+KOSIS 기반 예측)
- **지역 수**: 17개 행정구역

## 🔬 기술적 특징

### 프론트엔드
- **HTML5**: 시맨틱 마크업
- **CSS3**: Bootstrap 5 프레임워크 활용
- **JavaScript**: Vanilla JS로 구현된 인터랙티브 기능
- **차트 라이브러리**: Plotly.js
- **지도 라이브러리**: Leaflet.js

### 백엔드
- **Python 3.10+**: 데이터 처리 및 시뮬레이션
- **Pandas**: 데이터 분석 및 처리
- **NumPy**: 수치 계산
- **SQLite**: 데이터베이스 관리
- **Plotly**: 차트 생성

### 데이터 처리
- **실시간 API 연동**: KOSIS API를 통한 실시간 데이터 수집
- **시드 기반 랜덤**: 일관된 결과를 위한 시드 설정
- **실시간 계산**: 클라이언트 사이드에서 동적 계산
- **데이터 검증**: 안전한 데이터 처리 및 오류 처리

### 성능 최적화
- **지연 로딩**: 필요한 시점에 데이터 로드
- **메모리 효율성**: 불필요한 데이터 캐싱 방지
- **반응성**: 사용자 인터랙션에 즉시 반응

## 📈 사용 방법

### 1. 메인 대시보드
- **지역균형발전지수 확인**: 페이지 상단의 균형발전지수 카드에서 현재 점수와 수준 확인
- **재정자립도 분석**: 재정자립도 섹션에서 지역별 현황 및 격차 확인
- **BDS 트렌드 분석**: 체크박스를 통해 관심 지역 선택
- **상관관계 분석**: NAVIS vs BDS 상관관계 차트에서 연도별 변화 확인
- **지역별 지도 분석**: 지도에서 지역별 BDS 값 확인

### 2. 정책 시뮬레이터
- **선형 시뮬레이션**: 기본적인 정책 효과 예측
- **비선형 시뮬레이션**: 현실적인 정책 효과 모델링
- **정책 의사결정 시뮬레이션**: 종합적인 정책 평가 및 우선순위 결정
- **인터랙티브 시뮬레이터**: 사용자 정의 파라미터로 실시간 시뮬레이션

### 3. 데이터 출처 확인
- **네비게이션 바**: 데이터 출처 드롭다운 메뉴
- **각 섹션**: 데이터 출처 버튼으로 원본 데이터 확인

## 🎓 학술적 근거

### 주요 참고 문헌
- **Barro, R. J., & Sala-i-Martin, X. (1992).** "Convergence." Journal of Political Economy
- **OECD (2020).** "Regional Development Policy"
- **World Bank (2022).** "Local Government Fiscal Autonomy: International Comparisons"
- **IMF (2021).** "Fiscal Decentralization and Economic Growth"
- **한국개발연구원 (2022).** "지방자치단체 세원 다양화 정책 효과 분석"
- **국토연구원 (2021).** "지역균형발전 정책 효과성 평가"

### 모델링 근거
- **정책 효과 계수**: OECD, World Bank 연구 기반
- **비용-효과 분석**: IMF, KDI 연구 기반
- **리스크 평가**: Monte Carlo 시뮬레이션 기법
- **다기준 의사결정**: AHP(Analytic Hierarchy Process) 기법

## 🤝 기여 방법

1. 이슈 등록: 버그 리포트 또는 기능 요청
2. 포크 및 브랜치 생성
3. 코드 수정 및 테스트
4. 풀 리퀘스트 제출

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 등록해 주세요.

---

**개발 환경**: Python 3.10+, HTML5, CSS3, JavaScript  
**주요 라이브러리**: Bootstrap 5, Plotly.js, Leaflet.js, Pandas, NumPy  
**데이터 소스**: 한국은행 ECOS API, 통계청 KOSIS API  
**분석 기간**: 1997-2025년 (29년간)  
**분석 지역**: 17개 행정구역  
**최종 버전**: v3.0 (2025년 8월)