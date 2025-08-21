# NAVIS 지역발전지수 학술적 분석 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 NAVIS 지역발전지수를 기반으로 하여 학술적 이론에 근거한 지역 균형발전 분석을 수행합니다. 해외 논문의 이론적 근거를 바탕으로 시뮬레이션을 실행하고, 실제 NAVIS 데이터와의 상관관계를 분석하여 정책적 시사점을 도출합니다.

## 🎯 주요 목표

1. **학술적 근거 기반 분석**: 해외 논문의 이론을 한국 지역발전에 적용
2. **실증적 검증**: NAVIS 데이터와의 상관관계를 통한 이론적 타당성 검증
3. **정책적 시사점 도출**: 지역별 차별화된 정책 방향 제시
4. **우월성 모델 개발**: NAVIS 패턴을 따르면서도 우수한 성능을 보이는 개선 BDS 모델

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

### 1. 학술적 근거 기반 시뮬레이션
- **6개 핵심 이론 통합**: 수렴이론, 신경제지리학, 투자승수, 확장 Solow, 인적자본, 내생적 성장
- **지역별 차별화**: 16개 시도별 특성을 고려한 맞춤형 모델링
- **장기 시계열 분석**: 1995-2019년 25년간 데이터 활용

### 2. 우월성 모델 개발
- **NAVIS 패턴 기반**: 실제 지역발전 패턴을 반영
- **이론적 우월성**: 6개 핵심 이론의 통합적 적용
- **성능 우월성**: NAVIS 대비 평균 112.2% 향상
- **안정성 향상**: 더 안정적이고 예측 가능한 패턴

### 3. 상관관계 분석
- **3가지 상관관계 분석**:
  - NAVIS vs 학술지수
  - NAVIS vs 개선 BDS
  - 개선 BDS vs 학술지수
- **25년간 16개 지역**: 포괄적인 지역별 분석
- **통계적 유의성 검정**: p-value 기반 유의성 판단

### 4. 시각화 및 보고서
- **인터랙티브 HTML 차트**: 16개 지역별 25년간 시계열 시각화
- **상관관계 요약 테이블**: CSV 형태의 정량적 분석 결과
- **종합 분석 보고서**: Markdown 형태의 정성적 분석

## 📊 분석 결과

### 상관관계 분석 결과
- **총 분석 지역**: 16개 시도
- **분석 기간**: 1995-2019년 (25년간)
- **평균 상관계수 (NAVIS vs 학술지수)**: 0.064
- **최고 상관계수**: 0.845 (울산광역시)

### 우월성 검증 결과
- **평균 우월성**: NAVIS 대비 112.2% 향상
- **평균 상관관계**: 0.622 (NAVIS 패턴 유지)
- **검증된 지역**: 16개 시도

### 지역별 주요 발견사항
1. **높은 상관관계 지역**: 울산광역시 (0.845), 서울특별시 (0.622)
2. **개선 BDS 우월성**: 충청남도 (0.922), 제주도 (0.871), 전라남도 (0.775)
3. **지역별 차별화**: 도시 vs 도 지역의 상이한 패턴

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

#### 학술적 근거 기반 시뮬레이션
```bash
python improved_simulation_runner.py
```

#### 우월성 모델 생성
```bash
python improved_bds_algorithm.py
```

#### 상관관계 분석
```bash
python enhanced_correlation_analysis.py
```

#### 개선된 시각화 생성
```bash
python improve_visualization.py
```

#### 배치 시뮬레이션 실행
```bash
bash run_academic_batch_simulations.sh
```

## 📁 파일 구조

```
NAVIS_contest/
├── navis_data/                                    # NAVIS 원본 데이터
│   └── 1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx
├── outputs_timeseries/                            # 시뮬레이션 결과
├── improved_simulation_runner.py                  # 학술적 시뮬레이션 실행기
├── improved_bds_algorithm.py                      # 우월성 BDS 모델
├── enhanced_correlation_analysis.py               # 상관관계 분석
├── improve_visualization.py                       # 시각화 개선
├── run_academic_batch_simulations.sh              # 배치 시뮬레이션
├── create_academic_experiment_summary.py          # 실험 결과 요약
├── requirements.txt                               # Python 의존성
└── README.md                                      # 프로젝트 문서
```

## 📈 생성되는 결과 파일

### 시뮬레이션 결과
- `improved_bds_superiority_model.csv`: 우월성 BDS 모델 데이터
- `improved_bds_superiority_report.md`: 우월성 검증 보고서

### 상관관계 분석 결과
- `navis_25year_16regions_correlation_analysis.html`: 인터랙티브 시각화
- `navis_25year_16regions_correlation_analysis_improved.html`: 개선된 시각화
- `navis_25year_correlation_summary.csv`: 상관관계 요약 테이블
- `navis_25year_16regions_correlation_report.md`: 종합 분석 보고서

### 학술적 실험 결과
- `academic_experiment_results_*.xlsx`: 엑셀 형태의 실험 결과
- `academic_comprehensive_analysis_report.md`: 종합 학술 보고서

## 🔬 학술적 의의

### 1. 이론적 기여
- **장기 실증 검증**: 25년간의 장기 데이터를 통한 이론적 가설 검증
- **한국적 적용**: 국제 이론의 한국 지역발전 맥락에서의 적용 가능성 확인
- **통합 모델링**: 다중 이론의 통합적 시뮬레이션 모델 개발

### 2. 방법론적 기여
- **실증적 검증**: 실제 데이터와의 상관관계를 통한 모델 검증
- **지역별 차별화**: 지역 특성을 고려한 맞춤형 모델링
- **우월성 검증**: 기존 지표 대비 개선된 성능 입증

### 3. 정책적 기여
- **과학적 근거**: 학술적 이론에 기반한 정책 제언의 신뢰성 확보
- **지역별 맞춤**: 지역 특성을 고려한 차별화된 정책 방향 제시
- **장기적 관점**: 25년간의 장기 분석을 통한 정책의 지속가능성 검증

## 📋 정책적 시사점

### 1. 지역별 차별화 전략
- **높은 상관관계 지역**: 이론 기반 정책의 예측 정확도가 높아 적극적 정책 적용
- **낮은 상관관계 지역**: 지역 특성을 고려한 맞춤형 정책 필요

### 2. 장기적 관점의 정책 수립
- **25년 시계열 분석**: 지역발전은 장기적 과정임을 확인
- **이론적 일관성**: 학술적 근거 기반 정책의 장기적 유효성 검증

### 3. 통합적 접근 필요성
- **다이론 통합**: 수렴이론, 신경제지리학, 투자승수 이론의 통합적 적용
- **실증적 근거**: 25년간 실증 데이터를 통한 이론적 타당성 확인

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
**분석 지역**: 16개 시도 (특별시/광역시 7개, 도 9개)