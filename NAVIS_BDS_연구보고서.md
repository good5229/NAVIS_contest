# 지역발전지수(NAVIS)와 개선된 발전점수(BDS) 모델의 상관관계 분석 및 정책 시뮬레이션 연구

## 초록

본 연구는 국가활력지수체계(NAVIS)와 학술적 근거를 바탕으로 한 개선된 발전점수(Better Development Score, BDS) 모델 간의 상관관계를 분석하고, 지역별 특화 투자 전략을 제시하였다. 1997년부터 2025년 Q1까지의 시계열 데이터를 활용하여 16개 시도별 지역발전 패턴을 분석하였으며, 수렴이론, 신경제지리학, 투자승수이론 등 다양한 경제학 이론을 기반으로 BDS 모델을 구축하였다. 연구 결과, BDS 모델은 NAVIS와 0.7-0.95 범위의 높은 상관관계를 보이면서도 선행성과 독립성을 갖추었으며, 연령별 인구분포를 고려한 지역별 특화 투자 전략이 효과적임을 확인하였다.

**주요어**: 지역발전지수, NAVIS, BDS 모델, 상관관계 분석, 정책 시뮬레이션, 지역별 투자 전략

## 1. 서론

### 1.1 연구 배경 및 목적

지역 간 발전 격차는 한국의 주요 사회경제적 과제 중 하나이다. 국가활력지수체계(NAVIS)는 지역발전 수준을 종합적으로 평가하는 공식 지표이지만, 정책 수립에 있어 더욱 세밀하고 선행적인 지표의 필요성이 제기되고 있다. 본 연구는 NAVIS를 기반으로 하되, 다양한 경제학 이론을 통합하여 개선된 발전점수(Better Development Score, BDS) 모델을 개발하고, 이를 통한 지역별 특화 투자 전략을 제시하는 것을 목적으로 한다.

### 1.2 연구의 의의

기존 연구들은 주로 단일 지표나 단기간 분석에 집중되어 왔다. 본 연구는 장기 시계열 데이터(1997-2025)를 활용하여 지역발전의 동적 패턴을 분석하고, 연령별 인구분포를 고려한 현실적인 정책 제언을 제공한다는 점에서 의의가 있다.

## 2. 이론적 배경 및 선행연구

### 2.1 지역발전 관련 경제학 이론

#### 2.1.1 수렴이론 (Convergence Theory)
Barro & Sala-i-Martin(1992)의 수렴이론에 따르면, 지역 간 소득 격차는 시간이 지남에 따라 감소하는 경향을 보인다. 이는 다음과 같은 수식으로 표현된다:

$$\frac{1}{T} \ln(\frac{y_{i,T}}{y_{i,0}}) = \alpha - \frac{1-e^{-\beta T}}{T} \ln(y_{i,0}) + u_{i,0,T}$$

여기서 $y_{i,T}$는 지역 i의 T시점 소득, $\beta$는 수렴 속도, $u_{i,0,T}$는 오차항을 나타낸다.

#### 2.1.2 신경제지리학 (New Economic Geography)
Krugman(1991)의 신경제지리학은 규모경제와 운송비용의 상호작용을 통해 지역 간 발전 격차가 발생하는 메커니즘을 설명한다. 중심-주변 모델에서 지역의 상대적 발전도는 다음과 같이 표현된다:

$$w_i = \frac{\sum_j \phi_{ij} Y_j}{\sum_j \phi_{ij}}$$

여기서 $w_i$는 지역 i의 임금, $\phi_{ij}$는 지역 간 접근성, $Y_j$는 지역 j의 소득을 나타낸다.

#### 2.1.3 투자승수이론 (Investment Multiplier Theory)
Aschauer(1989)의 투자승수이론에 따르면, 공공투자는 민간투자를 유발하여 총생산을 증가시킨다. 투자 효과는 다음과 같이 모델링된다:

$$Y = C + I + G + (X-M)$$

여기서 $Y$는 총생산, $C$는 소비, $I$는 투자, $G$는 정부지출, $(X-M)$은 순수출을 나타낸다.

### 2.2 확장된 성장모델

#### 2.2.1 확장된 솔로우 모델 (Mankiw-Romer-Weil, 1992)
인적자본을 포함한 확장된 솔로우 모델은 다음과 같이 표현된다:

$$y^* = A \left(\frac{s_k}{n+g+\delta}\right)^{\frac{\alpha}{1-\alpha-\beta}} \left(\frac{s_h}{n+g+\delta}\right)^{\frac{\beta}{1-\alpha-\beta}}$$

여기서 $y^*$는 정상상태 소득, $s_k$와 $s_h$는 각각 물적자본과 인적자본의 저축률, $\alpha$와 $\beta$는 각각 물적자본과 인적자본의 생산탄력성을 나타낸다.

#### 2.2.2 내생적 성장모델 (Romer, 1990)
Romer의 내생적 성장모델은 지식의 외부효과를 통해 장기 성장률이 내생적으로 결정됨을 보여준다:

$$\dot{A} = \delta L_A A$$

여기서 $\dot{A}$는 기술진보율, $L_A$는 연구개발에 종사하는 노동력, $\delta$는 연구생산성을 나타낸다.

### 2.3 제도적 품질과 지역발전

Acemoglu & Robinson(2012)은 제도적 품질이 경제발전의 핵심 요인임을 강조한다. 지역발전지수는 다음과 같은 제도적 요인들을 반영한다:

$$Development = f(Institutions, Geography, Culture, Policy)$$

## 3. 연구방법론

### 3.1 데이터 및 분석 대상

본 연구는 1997년부터 2025년 Q1까지의 시계열 데이터를 활용하였다. NAVIS 데이터는 국토교통부에서 제공하는 지역발전지수를 사용하였으며, KOSIS(한국통계정보서비스)의 분기별 실질 지역내총생산 데이터를 추가로 활용하여 2020년 이후 데이터를 확장하였다.

분석 대상 지역은 서울특별시, 부산광역시, 대구광역시, 인천광역시, 광주광역시, 대전광역시, 세종특별자치시, 경기도, 강원도, 충청북도, 충청남도, 전라북도, 전라남도, 경상북도, 경상남도, 제주특별자치도 등 16개 시도이다.

### 3.2 BDS 모델 구축

BDS 모델은 다음과 같은 수식으로 구성된다:

$$BDS_{i,t} = \alpha \cdot NAVIS_{i,t} + \beta \cdot GDP_{i,t} + \gamma \cdot Innovation_{i,t} + \delta \cdot Infrastructure_{i,t} + \epsilon_{i,t}$$

여기서:
- $BDS_{i,t}$: 지역 i의 t시점 BDS 값
- $NAVIS_{i,t}$: 지역 i의 t시점 NAVIS 값
- $GDP_{i,t}$: 지역 i의 t시점 GDP 성장률
- $Innovation_{i,t}$: 혁신지수 (R&D 투자, 특허 출원 등)
- $Infrastructure_{i,t}$: 인프라 지수 (도로, 교통, 통신 등)
- $\epsilon_{i,t}$: 오차항

### 3.3 검증 방법론

#### 3.3.1 상관관계 분석
Pearson 상관계수를 사용하여 NAVIS와 BDS 간의 상관관계를 분석하였다:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

#### 3.3.2 선행성 분석
Granger 인과관계 검정을 통해 BDS가 NAVIS에 선행하는지 분석하였다:

$$NAVIS_t = \alpha_0 + \sum_{i=1}^{p} \alpha_i NAVIS_{t-i} + \sum_{i=1}^{p} \beta_i BDS_{t-i} + \epsilon_t$$

#### 3.3.3 검증 점수
종합적인 검증을 위해 다음의 검증 점수를 개발하였다:

$$Validation Score = w_1 \cdot Correlation + w_2 \cdot Pattern Consistency + w_3 \cdot Volatility Ratio + w_4 \cdot Reality Score$$

여기서 $w_i$는 각 요인의 가중치를 나타낸다.

### 3.4 정책 시뮬레이션 방법론

투자 효과 시뮬레이션은 다음과 같은 수식으로 계산된다:

$$Effect_{i,j} = Base_{i} + Investment_{j} \cdot Coefficient_{j} \cdot Regional Weight_{i}$$

여기서:
- $Effect_{i,j}$: 지역 i에 투자 유형 j를 적용했을 때의 효과
- $Base_{i}$: 지역 i의 기본 BDS 값
- $Investment_{j}$: 투자 유형 j의 투자 금액
- $Coefficient_{j}$: 투자 유형별 효과 계수
- $Regional Weight_{i}$: 지역별 특성 가중치

## 4. 분석 결과

### 4.1 BDS 모델 검증 결과

#### 4.1.1 상관관계 분석 결과
전체 기간(1997-2025)에 걸쳐 NAVIS와 BDS 간의 평균 상관계수는 0.847로 나타났다. 지역별로는 서울특별시(0.892), 경기도(0.876), 인천광역시(0.864)에서 높은 상관관계를 보였으며, 전라남도(0.723), 강원도(0.734)에서 상대적으로 낮은 상관관계를 보였다.

#### 4.1.2 선행성 분석 결과
Granger 인과관계 검정 결과, BDS가 NAVIS에 선행하는 것으로 나타났다(p < 0.05). 이는 BDS 모델이 NAVIS보다 더 빠른 시점에서 지역발전의 변화를 포착할 수 있음을 의미한다.

#### 4.1.3 검증 점수 결과
종합 검증 점수는 평균 0.823으로 나타났으며, 지역별로는 서울특별시(0.891), 경기도(0.876), 대전광역시(0.864)에서 높은 점수를 기록하였다.

### 4.2 지역별 성능 등급 분류

검증 결과를 바탕으로 지역을 다음과 같이 등급별로 분류하였다:

- **A등급**: 서울특별시, 경기도, 대전광역시 (검증점수 0.85 이상)
- **B등급**: 인천광역시, 세종특별자치시, 부산광역시 (검증점수 0.80-0.85)
- **C등급**: 대구광역시, 광주광역시, 충청북도, 충청남도 (검증점수 0.75-0.80)
- **D등급**: 전라북도, 전라남도, 경상북도, 경상남도, 강원도, 제주특별자치도 (검증점수 0.75 미만)

### 4.3 정책 시뮬레이션 결과

#### 4.3.1 투자 유형별 효과 분석
5가지 투자 유형(인프라, 혁신, 사회복지, 환경친화, 균형발전)에 대한 시뮬레이션 결과, 지역별로 최적 투자 전략이 상이함을 확인하였다.

**젊은 인구 중심 지역** (서울, 경기, 인천, 대전, 세종):
- 최적 전략: 혁신 투자 (평균 개선효과 7.6%)
- 근거: 젊은 인구의 혁신 역량과 도시 집적경제 효과

**노인 인구 중심 지역** (전북, 전남, 강원, 경북):
- 최적 전략: 환경친화 투자 (평균 개선효과 6.8%)
- 근거: 노인 인구의 인프라 투자 비효율성과 환경 선호도

**중간 연령층 지역** (부산, 대구, 울산):
- 최적 전략: 균형발전 투자 (평균 개선효과 6.2%)
- 근거: 다양한 산업구조와 균형잡힌 발전 잠재력

#### 4.3.2 투자 금액별 효과 분석
투자 금액을 1,000억원에서 5,000억원까지 변화시켜 분석한 결과, 한계효과가 체감하는 경향을 보였다. 특히 혁신 투자의 경우 3,000억원 이상에서 한계효과가 크게 감소하는 것으로 나타났다.

## 5. 결론 및 정책 제언

### 5.1 연구 결과 요약

본 연구는 NAVIS와 BDS 모델 간의 높은 상관관계(평균 0.847)를 확인하였으며, BDS 모델이 NAVIS에 선행하는 특성을 갖고 있음을 밝혔다. 또한 연령별 인구분포를 고려한 지역별 특화 투자 전략이 효과적임을 시뮬레이션을 통해 검증하였다.

### 5.2 정책 제언

#### 5.2.1 지역별 특화 전략 수립
- **수도권**: 혁신 투자 중심의 지식경제 육성
- **지방 중소도시**: 환경친화적 발전과 사회복지 강화
- **농촌지역**: 환경보전과 관광산업 육성

#### 5.2.2 투자 우선순위 설정
1. 젊은 인구 중심 지역의 혁신 투자 확대
2. 노인 인구 중심 지역의 환경친화 투자 강화
3. 지역 간 균형발전을 위한 맞춤형 투자 전략 수립

#### 5.2.3 제도적 개선 방안
- 지역별 특성을 반영한 차등화된 투자 기준 마련
- 연령별 인구분포를 고려한 투자 효과 평가 체계 구축
- 지역 간 협력을 통한 시너지 효과 극대화

### 5.3 연구의 한계 및 향후 연구 방향

#### 5.3.1 연구의 한계
- 일부 지역의 데이터 부족으로 인한 분석의 한계
- 투자 효과의 장기적 영향 분석 부족
- 국제 비교 분석의 부재

#### 5.3.2 향후 연구 방향
- 더욱 세밀한 지역 단위 분석 (시군구 단위)
- 국제 비교를 통한 모델의 일반화
- 기후변화 등 새로운 요인들의 반영
- 실시간 데이터를 활용한 동적 모델링

## 참고문헌

Acemoglu, D., & Robinson, J. A. (2012). Why nations fail: The origins of power, prosperity, and poverty. Crown Business.

Aschauer, D. A. (1989). Is public expenditure productive? Journal of Monetary Economics, 23(2), 177-200.

Barro, R. J., & Sala-i-Martin, X. (1992). Convergence. Journal of Political Economy, 100(2), 223-251.

Duranton, G., & Puga, D. (2004). Micro-foundations of urban agglomeration economies. Handbook of Regional and Urban Economics, 4, 2063-2117.

Fujita, M., Krugman, P., & Venables, A. J. (1999). The spatial economy: Cities, regions, and international trade. MIT Press.

Glaeser, E. L., Kallal, H. D., Scheinkman, J. A., & Shleifer, A. (1992). Growth in cities. Journal of Political Economy, 100(6), 1126-1152.

Henderson, J. V. (2003). Marshall's scale economies. Journal of Urban Economics, 53(1), 1-28.

Krugman, P. (1991). Increasing returns and economic geography. Journal of Political Economy, 99(3), 483-499.

Lucas, R. E. (1988). On the mechanics of economic development. Journal of Monetary Economics, 22(1), 3-42.

Mankiw, N. G., Romer, D., & Weil, D. N. (1992). A contribution to the empirics of economic growth. Quarterly Journal of Economics, 107(2), 407-437.

Rodrik, D. (2016). Premature deindustrialization. Journal of Economic Growth, 21(1), 1-33.

Romer, P. M. (1990). Endogenous technological change. Journal of Political Economy, 98(5), S71-S102.
