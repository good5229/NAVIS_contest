# NAVIS 지역발전지수 학술적 분석 프로젝트 - FINAL

## 📋 프로젝트 개요

이 프로젝트는 NAVIS 지역발전지수를 기반으로 하여 학술적 이론에 근거한 지역 균형발전 분석을 수행합니다. 해외 논문의 이론적 근거를 바탕으로 시뮬레이션을 실행하고, 실제 NAVIS 데이터와의 상관관계를 분석하여 정책적 시사점을 도출합니다.

## 🎯 주요 목표

1. **학술적 근거 기반 분석**: 해외 논문의 이론을 한국 지역발전에 적용
2. **실증적 검증**: NAVIS 데이터와의 상관관계를 통한 이론적 타당성 검증
3. **정책적 시사점 도출**: 지역별 차별화된 정책 방향 제시
4. **향상된 BDS 모델 개발**: NAVIS 패턴을 따르면서도 우수한 성능을 보이는 개선 BDS 모델

## 📚 이론적 근거

### 핵심 이론
1. **수렴이론** (Barro & Sala-i-Martin, 1992)
   - β-수렴계수: -0.02
   - 조건부 수렴 모델 적용

2. **신경제지리학** (Krugman, 1991)
   - 공간적 상호작용 효과: 0.1-0.2
   - 집적경제와 확산효과

3. **투자승수 이론** (Aschauer, 1989)
   - 투자승수: 1.5-2.1
   - 공공투자의 생산성 효과

4. **확장 Solow 모델** (Mankiw-Romer-Weil, 1992)
   - 물적자본 축적 효과
   - 인적자본 요소 추가

5. **인적자본 모델** (Lucas, 1988)
   - 인적자본 축적과 경제성장
   - 교육 투자 효과

6. **내생적 성장 모델** (Romer, 1990)
   - 내생적 기술진보
   - 연구개발 효율성

## 🚀 주요 기능

### 1. 향상된 BDS 모델 개발
- **NAVIS 패턴 기반**: 실제 지역발전 패턴을 정확히 반영
- **이론적 우월성**: 6개 핵심 이론의 통합적 적용
- **검증 시스템**: 상관관계, 선행성, 독립성, 변동성 검증
- **종합 검증 점수**: 0.282 (보완 지표 수준)

### 2. 종합 시각화 시스템
- **한국 지도 히트맵**: Geojson을 활용한 지역별 상관관계 시각화
- **Bootstrap 모달**: 상세한 설명과 도움말 시스템
- **인터랙티브 차트**: Plotly 기반의 동적 시각화
- **정책 시뮬레이션**: 지역별 특성을 반영한 투자 효과 분석

### 3. 정책 시뮬레이션
- **5가지 투자 유형**: 인프라, 혁신, 사회, 환경, 균형 투자
- **지역별 특성 반영**: 도시/도 지역별 차별화된 가중치 적용
- **투자 금액별 효과**: 지역별 개별 분석
- **최적 투자 전략**: 지역 특성을 고려한 맞춤형 전략 제시

## 📊 분석 결과

### 향상된 BDS 모델 검증 결과
- **총 지역**: 22개
- **선행성 우위**: 12개 지역 (54.5%)
- **평균 상관관계**: 0.934
- **평균 변동성 비율**: 1.147
- **종합 검증 점수**: 0.282

### 지역별 선행성 분석
**✅ 선행성 우위 지역 (12개)**:
- 서울특별시: 변동성 비율 1.622
- 대전광역시: 변동성 비율 1.512
- 인천광역시: 변동성 비율 1.456
- 대구광역시: 변동성 비율 1.434
- 경기도: 변동성 비율 1.326
- 경상북도: 변동성 비율 1.312
- 제주권: 변동성 비율 1.246
- 강원도: 변동성 비율 1.230
- 충청북도: 변동성 비율 1.212
- 광주광역시: 변동성 비율 1.190
- 충청남도: 변동성 비율 1.126
- 수도권: 변동성 비율 1.110

### 정책 시뮬레이션 결과
- **총 시나리오**: 110개
- **투자 유형**: 5가지 (인프라, 혁신, 사회, 환경, 균형)
- **지역별 특성 반영**: 도시/도 지역별 차별화된 전략
- **최적 투자 효과**: 지역별 맞춤형 투자 전략 제시

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

### 2. 주요 실행 파일

#### 향상된 BDS 모델 및 시각화
```bash
python enhanced_bds_visualization_final.py
```

## 📁 파일 구조

```
NAVIS_contest/
├── navis_data/                                    # NAVIS 원본 데이터
│   ├── 1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx
│   └── skorea-provinces-2018-geo.json            # 한국 지도 Geojson
├── enhanced_bds_visualization_final.py           # 향상된 BDS 모델 및 시각화
├── enhanced_bds_comprehensive_dashboard_final.html # 종합 대시보드
├── bds_policy_simulation_final.html              # 정책 시뮬레이션
├── bds_policy_simulation_results_final.csv       # 시뮬레이션 결과
├── requirements.txt                               # Python 의존성
├── .gitignore                                     # Git 제외 파일
├── LICENSE                                        # 라이선스
└── README.md                                      # 프로젝트 문서
```

## 📈 생성되는 결과 파일

### 시각화 결과
- `enhanced_bds_comprehensive_dashboard_final.html`: Bootstrap 모달이 포함된 종합 대시보드
- `bds_policy_simulation_final.html`: Bootstrap 모달이 포함된 정책 시뮬레이션

### 데이터 결과
- `bds_policy_simulation_results_final.csv`: 상세 시뮬레이션 결과

## 🔬 학술적 의의

### 1. 이론적 기여
- **실증적 검증**: NAVIS 데이터와의 높은 상관관계(0.934)를 통한 이론적 타당성 확인
- **선행성 입증**: 12개 지역에서 BDS가 NAVIS보다 선행적임을 확인
- **한국적 적용**: 국제 이론의 한국 지역발전 맥락에서의 적용 가능성 확인

### 2. 방법론적 기여
- **검증 시스템**: 상관관계, 선행성, 독립성, 변동성을 통합한 종합 검증 체계
- **지역별 차별화**: 지역 특성을 고려한 맞춤형 모델링
- **시각화 개선**: 한국 지도와 Bootstrap 모달을 활용한 직관적 시각화

### 3. 정책적 기여
- **과학적 근거**: 학술적 이론에 기반한 정책 제언의 신뢰성 확보
- **지역별 맞춤**: 지역 특성을 고려한 차별화된 정책 방향 제시
- **투자 시뮬레이션**: 실제 정책 결정에 활용 가능한 투자 효과 분석

## 📋 정책적 시사점

### 1. 지역별 차별화 전략
- **도시 지역** (특별시/광역시): 혁신, 사회 투자에 집중
- **도 지역**: 인프라, 환경 투자에 집중
- **기타 지역**: 균형 투자에 집중

### 2. 선행성 우위 지역 활용
- **12개 선행성 우위 지역**: BDS 지표를 활용한 선제적 정책 수립
- **예측 정확도**: 높은 상관관계를 통한 정책 효과 예측 가능

### 3. 투자 효율성 극대화
- **5가지 투자 유형**: 지역별 특성에 맞는 최적 투자 유형 선택
- **투자 금액별 효과**: 지역별 개별 분석을 통한 효율적 자원 배분

## 📚 참고 문헌

1. Barro, R. J., & Sala-i-Martin, X. (1992). Convergence. Journal of Political Economy, 100(2), 223-251.
2. Krugman, P. (1991). Geography and Trade. MIT Press.
3. Aschauer, D. A. (1989). Is public expenditure productive? Journal of Monetary Economics, 23(2), 177-200.
4. Mankiw, N. G., Romer, D., & Weil, D. N. (1992). A contribution to the empirics of economic growth. The Quarterly Journal of Economics, 107(2), 407-437.
5. Lucas, R. E. (1988). On the mechanics of economic development. Journal of Monetary Economics, 22(1), 3-42.
6. Romer, P. M. (1990). Endogenous technological change. Journal of Political Economy, 98(5), S71-S102.

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

**개발 환경**: Python 3.10+, macOS/Linux/Windows  
**주요 라이브러리**: Pandas, NumPy, Matplotlib, Plotly, SciPy, Scikit-learn  
**데이터 소스**: NAVIS 지역발전지수 (1995-2019)  
**분석 기간**: 25년간 (1995-2019)  
**분석 지역**: 22개 지역 (특별시/광역시, 도, 권역)  
**최종 버전**: FINAL (2024년 8월)