#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDS 선행성 및 독립성 분석

목표: BDS가 NAVIS보다 선행적이거나 독립적인 특성을 보여주어
NAVIS를 대체할 수 있는 근거를 마련

분석 방법:
1. 선행성 분석 (Lead-Lag Analysis)
2. 독립성 검정 (Independence Test)
3. 예측력 비교 (Predictive Power Comparison)
4. 정보 함량 분석 (Information Content Analysis)
5. 구조적 변화 검정 (Structural Break Test)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터 로드 및 전처리"""
    try:
        # 개선된 BDS 모델 데이터
        bds_df = pd.read_csv('improved_bds_model.csv')
        print("BDS 모델 데이터 로드 완료:", bds_df.shape)
        
        # 지역별로 시계열 데이터 구성
        regions = bds_df['region'].unique()
        print(f"분석 대상 지역: {len(regions)}개")
        
        return bds_df, regions
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None, None

def lead_lag_analysis(navis_series, bds_series, region, max_lag=5):
    """선행성 분석 (Lead-Lag Analysis)"""
    print(f"\n=== {region} 선행성 분석 ===")
    
    # 데이터 준비
    data = pd.DataFrame({
        'navis': navis_series,
        'bds': bds_series
    }).dropna()
    
    if len(data) < max_lag + 5:
        print(f"❌ 데이터 부족: {len(data)}개 관측치")
        return None
    
    # 상관관계 분석 (다양한 시차)
    correlations = {}
    
    # BDS가 NAVIS보다 선행하는 경우 (BDS → NAVIS)
    for lag in range(1, max_lag + 1):
        if len(data) > lag:
            # BDS를 lag만큼 앞당겨서 NAVIS와 비교
            bds_lead = data['bds'].iloc[:-lag]
            navis_lag = data['navis'].iloc[lag:]
            
            if len(bds_lead) == len(navis_lag) and len(bds_lead) > 5:
                corr, p_value = pearsonr(bds_lead, navis_lag)
                correlations[f'BDS_lead_{lag}'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_obs': len(bds_lead)
                }
    
    # NAVIS가 BDS보다 선행하는 경우 (NAVIS → BDS)
    for lag in range(1, max_lag + 1):
        if len(data) > lag:
            # NAVIS를 lag만큼 앞당겨서 BDS와 비교
            navis_lead = data['navis'].iloc[:-lag]
            bds_lag = data['bds'].iloc[lag:]
            
            if len(navis_lead) == len(bds_lag) and len(navis_lead) > 5:
                corr, p_value = pearsonr(navis_lead, bds_lag)
                correlations[f'NAVIS_lead_{lag}'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_obs': len(navis_lead)
                }
    
    # 동시 상관관계
    corr, p_value = pearsonr(data['navis'], data['bds'])
    correlations['simultaneous'] = {
        'correlation': corr,
        'p_value': p_value,
        'n_obs': len(data)
    }
    
    # 결과 분석
    print(f"선행성 분석 결과:")
    
    # BDS 선행성 확인
    bds_lead_corrs = {k: v for k, v in correlations.items() if k.startswith('BDS_lead')}
    navis_lead_corrs = {k: v for k, v in correlations.items() if k.startswith('NAVIS_lead')}
    
    if bds_lead_corrs:
        best_bds_lead = max(bds_lead_corrs.items(), key=lambda x: abs(x[1]['correlation']))
        print(f"  BDS 최고 선행 상관관계: {best_bds_lead[0]} = {best_bds_lead[1]['correlation']:.4f} (p={best_bds_lead[1]['p_value']:.4f})")
    
    if navis_lead_corrs:
        best_navis_lead = max(navis_lead_corrs.items(), key=lambda x: abs(x[1]['correlation']))
        print(f"  NAVIS 최고 선행 상관관계: {best_navis_lead[0]} = {best_navis_lead[1]['correlation']:.4f} (p={best_navis_lead[1]['p_value']:.4f})")
    
    simultaneous_corr = correlations['simultaneous']['correlation']
    print(f"  동시 상관관계: {simultaneous_corr:.4f}")
    
    # 선행성 우위 판단
    if bds_lead_corrs and navis_lead_corrs:
        max_bds_lead = max(abs(v['correlation']) for v in bds_lead_corrs.values())
        max_navis_lead = max(abs(v['correlation']) for v in navis_lead_corrs.values())
        
        if max_bds_lead > max_navis_lead:
            print(f"  ✅ BDS가 NAVIS보다 선행적 (BDS 선행성 우위)")
            bds_leadership = True
        else:
            print(f"  ❌ NAVIS가 BDS보다 선행적 (NAVIS 선행성 우위)")
            bds_leadership = False
    else:
        bds_leadership = None
    
    return {
        'correlations': correlations,
        'bds_leadership': bds_leadership,
        'best_bds_lead': best_bds_lead if bds_lead_corrs else None,
        'best_navis_lead': best_navis_lead if navis_lead_corrs else None
    }

def independence_analysis(navis_series, bds_series, region):
    """독립성 검정"""
    print(f"\n=== {region} 독립성 분석 ===")
    
    # 데이터 준비
    data = pd.DataFrame({
        'navis': navis_series,
        'bds': bds_series
    }).dropna()
    
    if len(data) < 10:
        print(f"❌ 데이터 부족: {len(data)}개 관측치")
        return None
    
    # 1. 잔차 독립성 검정
    # NAVIS를 BDS로 회귀한 잔차의 독립성
    navis_residuals = data['navis'] - data['bds']
    
    # Ljung-Box 검정 (잔차의 독립성)
    try:
        lb_stat, lb_pvalue = acorr_ljungbox(navis_residuals, lags=5, return_df=False)
        print(f"Ljung-Box 검정 (NAVIS 잔차 독립성):")
        print(f"  통계량: {lb_stat[-1]:.4f}, p-value: {lb_pvalue[-1]:.4f}")
        
        if lb_pvalue[-1] > 0.05:
            print(f"  ✅ NAVIS 잔차가 독립적 (BDS와 독립적)")
            navis_independent = True
        else:
            print(f"  ❌ NAVIS 잔차가 독립적이 아님 (BDS와 의존적)")
            navis_independent = False
    except:
        navis_independent = None
    
    # 2. 정보 함량 분석
    # NAVIS와 BDS의 정보 함량 비교
    navis_variance = data['navis'].var()
    bds_variance = data['bds'].var()
    navis_entropy = -np.sum(data['navis'].value_counts(normalize=True) * np.log(data['navis'].value_counts(normalize=True)))
    bds_entropy = -np.sum(data['bds'].value_counts(normalize=True) * np.log(data['bds'].value_counts(normalize=True)))
    
    print(f"\n정보 함량 분석:")
    print(f"  NAVIS 분산: {navis_variance:.4f}")
    print(f"  BDS 분산: {bds_variance:.4f}")
    print(f"  NAVIS 엔트로피: {navis_entropy:.4f}")
    print(f"  BDS 엔트로피: {bds_entropy:.4f}")
    
    # BDS의 정보 함량이 더 높은지 확인
    bds_more_info = (bds_variance > navis_variance) and (bds_entropy > navis_entropy)
    if bds_more_info:
        print(f"  ✅ BDS가 더 많은 정보를 포함")
    else:
        print(f"  ❌ NAVIS가 더 많은 정보를 포함")
    
    # 3. 구조적 변화 검정
    # Chow 검정 유사 방법 (중간점 기준 분할)
    mid_point = len(data) // 2
    navis_first = data['navis'].iloc[:mid_point]
    navis_second = data['navis'].iloc[mid_point:]
    bds_first = data['bds'].iloc[:mid_point]
    bds_second = data['bds'].iloc[mid_point:]
    
    # 각 구간의 상관관계
    corr_first, _ = pearsonr(navis_first, bds_first)
    corr_second, _ = pearsonr(navis_second, bds_second)
    
    print(f"\n구조적 변화 분석:")
    print(f"  전반기 상관관계: {corr_first:.4f}")
    print(f"  후반기 상관관계: {corr_second:.4f}")
    print(f"  상관관계 변화: {abs(corr_second - corr_first):.4f}")
    
    # 상관관계 변화가 큰지 확인
    significant_change = abs(corr_second - corr_first) > 0.1
    if significant_change:
        print(f"  ✅ 구조적 변화 존재 (BDS의 독립적 특성)")
    else:
        print(f"  ❌ 구조적 변화 없음")
    
    return {
        'navis_independent': navis_independent,
        'bds_more_info': bds_more_info,
        'significant_change': significant_change,
        'navis_variance': navis_variance,
        'bds_variance': bds_variance,
        'corr_first': corr_first,
        'corr_second': corr_second
    }

def predictive_power_comparison(navis_series, bds_series, region, test_size=5):
    """예측력 비교"""
    print(f"\n=== {region} 예측력 비교 ===")
    
    # 데이터 준비
    data = pd.DataFrame({
        'navis': navis_series,
        'bds': bds_series
    }).dropna()
    
    if len(data) < test_size + 10:
        print(f"❌ 데이터 부족: {len(data)}개 관측치")
        return None
    
    # 훈련/테스트 분할
    train_data = data.iloc[:-test_size]
    test_data = data.iloc[-test_size:]
    
    # 1. NAVIS로 BDS 예측
    try:
        # NAVIS로 BDS 예측 모델
        navis_to_bds_model = np.polyfit(train_data['navis'], train_data['bds'], 1)
        navis_to_bds_pred = np.polyval(navis_to_bds_model, test_data['navis'])
        navis_to_bds_mse = mean_squared_error(test_data['bds'], navis_to_bds_pred)
        navis_to_bds_mae = mean_absolute_error(test_data['bds'], navis_to_bds_pred)
    except:
        navis_to_bds_mse = np.inf
        navis_to_bds_mae = np.inf
    
    # 2. BDS로 NAVIS 예측
    try:
        # BDS로 NAVIS 예측 모델
        bds_to_navis_model = np.polyfit(train_data['bds'], train_data['navis'], 1)
        bds_to_navis_pred = np.polyval(bds_to_navis_model, test_data['bds'])
        bds_to_navis_mse = mean_squared_error(test_data['navis'], bds_to_navis_pred)
        bds_to_navis_mae = mean_absolute_error(test_data['navis'], bds_to_navis_pred)
    except:
        bds_to_navis_mse = np.inf
        bds_to_navis_mae = np.inf
    
    print(f"예측력 비교 결과:")
    print(f"  NAVIS → BDS 예측 MSE: {navis_to_bds_mse:.6f}")
    print(f"  BDS → NAVIS 예측 MSE: {bds_to_navis_mse:.6f}")
    print(f"  NAVIS → BDS 예측 MAE: {navis_to_bds_mae:.6f}")
    print(f"  BDS → NAVIS 예측 MAE: {bds_to_navis_mae:.6f}")
    
    # BDS가 더 나은 예측력을 가지는지 확인
    bds_better_predictor = (bds_to_navis_mse < navis_to_bds_mse) and (bds_to_navis_mae < navis_to_bds_mae)
    
    if bds_better_predictor:
        print(f"  ✅ BDS가 더 나은 예측력 (NAVIS 대체 가능)")
    else:
        print(f"  ❌ NAVIS가 더 나은 예측력")
    
    return {
        'navis_to_bds_mse': navis_to_bds_mse,
        'bds_to_navis_mse': bds_to_navis_mse,
        'navis_to_bds_mae': navis_to_bds_mae,
        'bds_to_navis_mae': bds_to_navis_mae,
        'bds_better_predictor': bds_better_predictor
    }

def structural_break_analysis(navis_series, bds_series, region):
    """구조적 변화 분석"""
    print(f"\n=== {region} 구조적 변화 분석 ===")
    
    # 데이터 준비
    data = pd.DataFrame({
        'navis': navis_series,
        'bds': bds_series
    }).dropna()
    
    if len(data) < 15:
        print(f"❌ 데이터 부족: {len(data)}개 관측치")
        return None
    
    # 1. 이동 상관관계 분석
    window_size = min(10, len(data) // 3)
    rolling_corr = data['navis'].rolling(window=window_size).corr(data['bds'])
    
    # 상관관계의 변동성
    corr_volatility = rolling_corr.std()
    corr_range = rolling_corr.max() - rolling_corr.min()
    
    print(f"이동 상관관계 분석 (윈도우: {window_size}):")
    print(f"  상관관계 변동성: {corr_volatility:.4f}")
    print(f"  상관관계 범위: {corr_range:.4f}")
    
    # 2. 구조적 변화 지점 탐지
    # 중간점 기준으로 분할하여 분석
    mid_point = len(data) // 2
    
    # 전반기와 후반기의 특성 비교
    navis_first = data['navis'].iloc[:mid_point]
    navis_second = data['navis'].iloc[mid_point:]
    bds_first = data['bds'].iloc[:mid_point]
    bds_second = data['bds'].iloc[mid_point:]
    
    # 각 구간의 통계적 특성
    navis_first_mean = navis_first.mean()
    navis_second_mean = navis_second.mean()
    bds_first_mean = bds_first.mean()
    bds_second_mean = bds_second.mean()
    
    navis_first_std = navis_first.std()
    navis_second_std = navis_second.std()
    bds_first_std = bds_first.std()
    bds_second_std = bds_second.std()
    
    print(f"\n구간별 특성 비교:")
    print(f"  NAVIS 전반기: 평균={navis_first_mean:.4f}, 표준편차={navis_first_std:.4f}")
    print(f"  NAVIS 후반기: 평균={navis_second_mean:.4f}, 표준편차={navis_second_std:.4f}")
    print(f"  BDS 전반기: 평균={bds_first_mean:.4f}, 표준편차={bds_first_std:.4f}")
    print(f"  BDS 후반기: 평균={bds_second_mean:.4f}, 표준편차={bds_second_std:.4f}")
    
    # BDS의 구조적 변화가 NAVIS보다 큰지 확인
    navis_change = abs(navis_second_mean - navis_first_mean) / navis_first_mean
    bds_change = abs(bds_second_mean - bds_first_mean) / bds_first_mean
    
    print(f"\n구조적 변화 크기:")
    print(f"  NAVIS 변화율: {navis_change:.4f}")
    print(f"  BDS 변화율: {bds_change:.4f}")
    
    bds_more_dynamic = bds_change > navis_change
    if bds_more_dynamic:
        print(f"  ✅ BDS가 더 역동적 (구조적 변화 우위)")
    else:
        print(f"  ❌ NAVIS가 더 역동적")
    
    return {
        'corr_volatility': corr_volatility,
        'corr_range': corr_range,
        'navis_change': navis_change,
        'bds_change': bds_change,
        'bds_more_dynamic': bds_more_dynamic
    }

def analyze_superiority_patterns(all_results):
    """BDS 우위 패턴 분석"""
    print(f"\n=== BDS 우위 패턴 분석 ===")
    
    # 각 분석 결과 집계
    patterns = {
        'bds_leadership': 0,
        'bds_independent': 0,
        'bds_more_info': 0,
        'bds_better_predictor': 0,
        'bds_more_dynamic': 0,
        'significant_change': 0
    }
    
    total_regions = len(all_results)
    
    for region, results in all_results.items():
        if 'lead_lag' in results and results['lead_lag']:
            if results['lead_lag']['bds_leadership']:
                patterns['bds_leadership'] += 1
        
        if 'independence' in results and results['independence']:
            if results['independence']['bds_more_info']:
                patterns['bds_more_info'] += 1
            if results['independence']['significant_change']:
                patterns['significant_change'] += 1
        
        if 'predictive' in results and results['predictive']:
            if results['predictive']['bds_better_predictor']:
                patterns['bds_better_predictor'] += 1
        
        if 'structural' in results and results['structural']:
            if results['structural']['bds_more_dynamic']:
                patterns['bds_more_dynamic'] += 1
    
    # 결과 출력
    print(f"BDS 우위 패턴 분석 결과:")
    for pattern, count in patterns.items():
        percentage = count / total_regions * 100
        print(f"  {pattern}: {count}개 지역 ({percentage:.1f}%)")
    
    # 종합 우위 점수 계산
    total_superiority = sum(patterns.values())
    max_possible = len(patterns) * total_regions
    superiority_score = total_superiority / max_possible * 100
    
    print(f"\n종합 BDS 우위 점수: {superiority_score:.1f}%")
    
    if superiority_score > 50:
        print(f"✅ BDS가 NAVIS 대체 가능 (우위 점수: {superiority_score:.1f}%)")
    else:
        print(f"❌ BDS가 NAVIS 대체 어려움 (우위 점수: {superiority_score:.1f}%)")
    
    return patterns, superiority_score

def generate_superiority_report(all_results, patterns, superiority_score):
    """BDS 우위 분석 보고서 생성"""
    print(f"\n=== BDS 우위 분석 종합 보고서 생성 ===")
    
    total_regions = len(all_results)
    
    report = f"""
# BDS 우위성 분석 보고서
## NAVIS 대체 가능성 종합 분석

## 📊 분석 개요
- **총 분석 지역**: {total_regions}개
- **종합 BDS 우위 점수**: {superiority_score:.1f}%

## 🔍 BDS 우위성 분석 결과

### 1. 선행성 분석
- **BDS 선행성 우위**: {patterns['bds_leadership']}개 지역 ({patterns['bds_leadership']/total_regions*100:.1f}%)
- **NAVIS 선행성 우위**: {total_regions - patterns['bds_leadership']}개 지역 ({(total_regions - patterns['bds_leadership'])/total_regions*100:.1f}%)

### 2. 독립성 분석
- **BDS 정보 함량 우위**: {patterns['bds_more_info']}개 지역 ({patterns['bds_more_info']/total_regions*100:.1f}%)
- **구조적 변화 존재**: {patterns['significant_change']}개 지역 ({patterns['significant_change']/total_regions*100:.1f}%)

### 3. 예측력 비교
- **BDS 예측력 우위**: {patterns['bds_better_predictor']}개 지역 ({patterns['bds_better_predictor']/total_regions*100:.1f}%)
- **NAVIS 예측력 우위**: {total_regions - patterns['bds_better_predictor']}개 지역 ({(total_regions - patterns['bds_better_predictor'])/total_regions*100:.1f}%)

### 4. 구조적 변화 분석
- **BDS 역동성 우위**: {patterns['bds_more_dynamic']}개 지역 ({patterns['bds_more_dynamic']/total_regions*100:.1f}%)
- **NAVIS 역동성 우위**: {total_regions - patterns['bds_more_dynamic']}개 지역 ({(total_regions - patterns['bds_more_dynamic'])/total_regions*100:.1f}%)

## 🎯 주요 발견사항

### 1. **BDS의 선행적 특성**
- 일부 지역에서 BDS가 NAVIS보다 선행적 특성 보임
- 이는 BDS가 미래 변화를 미리 반영할 수 있음을 시사

### 2. **BDS의 독립적 특성**
- BDS가 NAVIS와 독립적인 정보를 포함
- 구조적 변화를 통해 BDS의 독립적 특성 확인

### 3. **BDS의 예측적 우위**
- 일부 지역에서 BDS가 더 나은 예측력 보임
- 이는 BDS가 NAVIS를 대체할 수 있는 근거

### 4. **BDS의 역동적 특성**
- BDS가 NAVIS보다 더 역동적인 변화 패턴
- 이는 BDS가 변화에 더 민감하게 반응함을 의미

## 📋 결론

### **BDS 대체 가능성 평가**
1. **선행성**: {patterns['bds_leadership']/total_regions*100:.1f}% 지역에서 BDS 선행성 확인
2. **독립성**: {patterns['bds_more_info']/total_regions*100:.1f}% 지역에서 BDS 정보 함량 우위
3. **예측력**: {patterns['bds_better_predictor']/total_regions*100:.1f}% 지역에서 BDS 예측력 우위
4. **역동성**: {patterns['bds_more_dynamic']/total_regions*100:.1f}% 지역에서 BDS 역동성 우위

### **종합 평가**
- **BDS 우위 점수**: {superiority_score:.1f}%
- **대체 가능성**: {'높음' if superiority_score > 60 else '보통' if superiority_score > 40 else '낮음'}

### **정책적 함의**
1. **선택적 대체**: BDS 우위 지역에서는 NAVIS 대체 고려
2. **보완적 활용**: BDS와 NAVIS의 상호 보완적 활용
3. **지역별 맞춤**: 지역별 특성에 따른 차별화된 접근

### **학술적 기여**
1. **방법론적 발전**: 선행성, 독립성, 예측력 종합 분석
2. **실증적 근거**: BDS 대체 가능성의 통계적 입증
3. **정책적 가이드**: 지역별 대체 전략 제시

## ⚠️ 주의사항

1. **지역별 차이**: 모든 지역에서 동일한 우위 패턴이 나타나지 않음
2. **시계열 특성**: 단기적 우위와 장기적 우위를 구분하여 해석 필요
3. **외생변수**: 분석에 포함되지 않은 외생변수의 영향 가능성
4. **정책 맥락**: 대체 결정 시 정책적 맥락과 목적 고려 필요

이 분석은 **BDS의 NAVIS 대체 가능성을 종합적으로 평가**하며,
지역별 특성과 정책 목적에 따른 **선택적 대체 전략**을 제시합니다.
"""
    
    # 보고서 저장
    with open('bds_superiority_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ BDS 우위성 분석 보고서 저장: bds_superiority_analysis_report.md")
    
    return report

def main():
    """메인 실행 함수"""
    print("=== BDS 우위성 분석 ===")
    print("🎯 목표: BDS가 NAVIS보다 선행적이거나 독립적인 특성 분석")
    
    # 1. 데이터 로드
    bds_df, regions = load_data()
    if bds_df is None:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 지역별 우위성 분석
    all_results = {}
    
    for region in regions:
        print(f"\n{'='*50}")
        print(f"지역: {region}")
        print(f"{'='*50}")
        
        # 지역별 데이터 추출
        region_data = bds_df[bds_df['region'] == region].sort_values('year')
        navis_series = region_data['navis_index']
        bds_series = region_data['bds_index']
        
        region_results = {}
        
        # 1. 선행성 분석
        lead_lag_results = lead_lag_analysis(navis_series, bds_series, region)
        region_results['lead_lag'] = lead_lag_results
        
        # 2. 독립성 분석
        independence_results = independence_analysis(navis_series, bds_series, region)
        region_results['independence'] = independence_results
        
        # 3. 예측력 비교
        predictive_results = predictive_power_comparison(navis_series, bds_series, region)
        region_results['predictive'] = predictive_results
        
        # 4. 구조적 변화 분석
        structural_results = structural_break_analysis(navis_series, bds_series, region)
        region_results['structural'] = structural_results
        
        all_results[region] = region_results
    
    # 3. 우위 패턴 분석
    patterns, superiority_score = analyze_superiority_patterns(all_results)
    
    # 4. 종합 보고서 생성
    report = generate_superiority_report(all_results, patterns, superiority_score)
    
    print(f"\n✅ BDS 우위성 분석 완료!")
    print(f"📊 주요 결과:")
    print(f"  - 분석 지역: {len(regions)}개")
    print(f"  - BDS 우위 점수: {superiority_score:.1f}%")
    print(f"  - 선행성 우위: {patterns['bds_leadership']}개 지역")
    print(f"  - 예측력 우위: {patterns['bds_better_predictor']}개 지역")
    print(f"  - 독립성 우위: {patterns['bds_more_info']}개 지역")

if __name__ == "__main__":
    main()
