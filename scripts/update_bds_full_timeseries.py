#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 시계열 ECOS 데이터로 BDS 재계산 (1997-2025년)
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def update_bds_full_timeseries():
    """전체 시계열 ECOS 데이터로 BDS 재계산"""
    
    print("🚀 전체 시계열 BDS 재계산 시작 (1997-2025년)")
    
    # 1. 데이터 로드
    print("\n📊 데이터 로드 중...")
    
    # KOSIS 재정자립도 데이터 (2001-2025년)
    fiscal_file = Path('data/fiscal_autonomy/kosis_fiscal_autonomy_data.csv')
    fiscal_df = pd.read_csv(fiscal_file)
    fiscal_df = fiscal_df[fiscal_df['region'] != '전국']
    print(f"  ✅ 재정자립도 데이터: {len(fiscal_df)} 건 ({fiscal_df['year'].min()}-{fiscal_df['year'].max()}년)")
    
    # KOSIS GDP 데이터 (2023년만)
    gdp_file = Path('data/kosis/kosis_gdp_data_2023.csv')
    gdp_df = pd.read_csv(gdp_file)
    print(f"  ✅ GDP 데이터: {len(gdp_df)} 건 (2023년)")
    
    # ECOS 데이터 (1997-2025년)
    ecos_cpi_file = Path('data/ecos/ecos_cpi_regional_timeseries.csv')
    ecos_mfg_file = Path('data/ecos/ecos_manufacturing_regional_timeseries.csv')
    
    if not ecos_cpi_file.exists() or not ecos_mfg_file.exists():
        print("❌ ECOS 시계열 데이터가 없습니다.")
        return False
    
    ecos_cpi_df = pd.read_csv(ecos_cpi_file)
    ecos_mfg_df = pd.read_csv(ecos_mfg_file)
    
    print(f"  ✅ ECOS CPI 데이터: {len(ecos_cpi_df)} 건 ({ecos_cpi_df['year'].min():.0f}-{ecos_cpi_df['year'].max():.0f}년)")
    print(f"  ✅ ECOS 제조업 데이터: {len(ecos_mfg_df)} 건 ({ecos_mfg_df['year'].min():.0f}-{ecos_mfg_df['year'].max():.0f}년)")
    
    # 2. 연도별 BDS 계산
    print("\n🔄 연도별 BDS 계산 중...")
    
    # 공통 연도 찾기 (재정자립도, ECOS 데이터 교집합)
    fiscal_years = set(fiscal_df['year'].unique())
    ecos_years = set(ecos_cpi_df['year'].astype(int).unique())
    common_years = sorted(fiscal_years & ecos_years)
    
    print(f"  📅 공통 연도: {common_years[0]}-{common_years[-1]} ({len(common_years)}년)")
    
    # 결과 저장용
    all_bds_results = []
    yearly_bds_baselines = {}
    
    def normalize(series):
        """Min-Max 정규화"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    # 연도별 계산
    for year in common_years:
        print(f"  📊 {year}년 BDS 계산 중...", end=" ")
        
        # 해당 연도 데이터 추출
        fiscal_year = fiscal_df[fiscal_df['year'] == year].copy()
        ecos_cpi_year = ecos_cpi_df[ecos_cpi_df['year'] == year].copy()
        ecos_mfg_year = ecos_mfg_df[ecos_mfg_df['year'] == year].copy()
        
        # GDP는 2023년 데이터를 모든 연도에 사용 (가장 최신 데이터)
        gdp_year = gdp_df.copy()
        
        # 데이터 병합
        merged = fiscal_year.merge(gdp_year, left_on='region', right_on='region', how='inner')
        merged = merged.merge(ecos_cpi_year[['region', 'value']].rename(columns={'value': 'cpi'}), 
                             on='region', how='left')
        merged = merged.merge(ecos_mfg_year[['region', 'value']].rename(columns={'value': 'manufacturing_index'}), 
                             on='region', how='left')
        
        if len(merged) < 10:  # 데이터가 너무 적으면 스킵
            print("❌ 데이터 부족")
            continue
        
        # 정규화
        merged['fiscal_norm'] = normalize(merged['fiscal_autonomy_ratio'])
        merged['gdp_norm'] = normalize(merged['gdp_2023'])
        merged['cpi_norm'] = 1 - normalize(merged['cpi'])  # 역정규화 (낮을수록 좋음)
        merged['manufacturing_norm'] = normalize(merged['manufacturing_index'])
        
        # BDS 계산 (가중 평균)
        weights = {
            'gdp': 0.30,
            'fiscal': 0.25,
            'cpi': 0.20,
            'manufacturing': 0.25
        }
        
        merged['bds_score'] = (
            merged['gdp_norm'] * weights['gdp'] +
            merged['fiscal_norm'] * weights['fiscal'] +
            merged['cpi_norm'] * weights['cpi'] +
            merged['manufacturing_norm'] * weights['manufacturing']
        ) * 10.0
        
        # 결과 저장
        for _, row in merged.iterrows():
            all_bds_results.append({
                'year': year,
                'region': row['region'],
                'bds_score': row['bds_score'],
                'fiscal_autonomy_ratio': row['fiscal_autonomy_ratio'],
                'gdp_2023': row['gdp_2023'],
                'cpi': row['cpi'],
                'manufacturing_index': row['manufacturing_index']
            })
        
        # 연도별 베이스라인 저장
        yearly_bds_baselines[str(year)] = {
            row['region']: float(row['bds_score']) for _, row in merged.iterrows()
        }
        
        print(f"✅ {len(merged)}개 지역")
    
    # 3. 결과 저장
    print(f"\n💾 결과 저장 중...")
    
    if all_bds_results:
        # 전체 결과 CSV로 저장
        all_results_df = pd.DataFrame(all_bds_results)
        all_results_df.to_csv(Path('data/bds/bds_full_timeseries_results.csv'), index=False)
        print(f"  ✅ 전체 BDS 시계열 결과: {len(all_results_df)} 건")
        
        # 최신 연도 베이스라인 JSON으로 저장
        latest_year = max([int(k) for k in yearly_bds_baselines.keys()])
        bds_result = {
            "latestYear": int(latest_year),
            "baselines": yearly_bds_baselines[str(latest_year)],
            "data_sources": {
                "kosis_fiscal": "재정자립도 (25%)",
                "kosis_gdp": "지역내총생산 (30%)",
                "ecos_cpi": "소비자물가지수 (20%)",
                "ecos_manufacturing": "제조업생산지수 (25%)"
            },
            "timeseries_years": f"{common_years[0]}-{common_years[-1]}",
            "total_data_points": len(all_results_df),
            "updated": "2025-09-29"
        }
        
        bds_file = Path('data/bds/bds_baseline.json')
        with open(bds_file, 'w', encoding='utf-8') as f:
            json.dump(bds_result, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 최신 BDS 베이스라인 업데이트: {latest_year}년")
        
        # 연도별 베이스라인 전체 저장
        yearly_file = Path('data/bds/bds_yearly_baselines.json')
        with open(yearly_file, 'w', encoding='utf-8') as f:
            json.dump(yearly_bds_baselines, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 연도별 BDS 베이스라인: {len(yearly_bds_baselines)}년치")
        
        # 4. 통계 요약
        print(f"\n📊 BDS 시계열 통계 요약:")
        
        # 최고/최저 지역
        latest_data = all_results_df[all_results_df['year'] == latest_year].sort_values('bds_score', ascending=False)
        print(f"  🏆 {latest_year}년 상위 5개 지역:")
        for i, (_, row) in enumerate(latest_data.head(5).iterrows(), 1):
            print(f"    {i}. {row['region']}: {row['bds_score']:.2f}점")
        
        print(f"  📉 {latest_year}년 하위 5개 지역:")
        for i, (_, row) in enumerate(latest_data.tail(5).iterrows(), 1):
            print(f"    {i}. {row['region']}: {row['bds_score']:.2f}점")
        
        # 시계열 트렌드
        yearly_avg = all_results_df.groupby('year')['bds_score'].mean()
        print(f"\n  📈 연도별 전국 평균 BDS:")
        print(f"    • 시작: {yearly_avg.iloc[0]:.2f}점 ({yearly_avg.index[0]}년)")
        print(f"    • 최신: {yearly_avg.iloc[-1]:.2f}점 ({yearly_avg.index[-1]}년)")
        print(f"    • 변화: {yearly_avg.iloc[-1] - yearly_avg.iloc[0]:+.2f}점")
        
        return True
    
    return False

if __name__ == "__main__":
    success = update_bds_full_timeseries()
    if success:
        print("\n🎉 전체 시계열 BDS 재계산 완료!")
    else:
        print("\n❌ BDS 재계산 실패")
