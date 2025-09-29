#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECOS 데이터를 BDS 계산에 맞게 처리
"""

import pandas as pd
from pathlib import Path

def process_ecos_data():
    """ECOS 데이터를 연도별로 집계하여 BDS 계산에 사용할 수 있도록 처리"""
    
    ecos_dir = Path('data/ecos')
    
    # 1. 소비자물가지수 처리
    cpi_file = ecos_dir / 'ecos_cpi_national.csv'
    if cpi_file.exists():
        print("소비자물가지수 데이터 처리 중...")
        cpi_df = pd.read_csv(cpi_file)
        
        # 연도별 평균 계산
        cpi_yearly = cpi_df.groupby('year')['value'].mean().reset_index()
        cpi_yearly['region'] = '전국'
        cpi_yearly = cpi_yearly[['region', 'year', 'value']]
        
        # 지역별로 동일한 값 복제 (전국 데이터를 모든 지역에 적용)
        regions = [
            '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', 
            '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도', 
            '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', 
            '경상남도', '제주특별자치도'
        ]
        
        cpi_regional = []
        for _, row in cpi_yearly.iterrows():
            for region in regions:
                cpi_regional.append({
                    'region': region,
                    'year': row['year'],
                    'value': row['value']
                })
        
        cpi_regional_df = pd.DataFrame(cpi_regional)
        cpi_regional_df.to_csv(ecos_dir / 'ecos_cpi.csv', index=False)
        print(f"소비자물가지수 지역별 데이터 생성: {len(cpi_regional_df)} 건")
    
    # 2. 제조업생산지수 처리
    mfg_file = ecos_dir / 'ecos_manufacturing_production_index.csv'
    if mfg_file.exists():
        print("제조업생산지수 데이터 처리 중...")
        mfg_df = pd.read_csv(mfg_file)
        
        # 연도별 평균 계산
        mfg_yearly = mfg_df.groupby('year')['value'].mean().reset_index()
        mfg_yearly['region'] = '전국'
        mfg_yearly = mfg_yearly[['region', 'year', 'value']]
        
        # 지역별로 동일한 값 복제
        mfg_regional = []
        for _, row in mfg_yearly.iterrows():
            for region in regions:
                mfg_regional.append({
                    'region': region,
                    'year': row['year'],
                    'value': row['value']
                })
        
        mfg_regional_df = pd.DataFrame(mfg_regional)
        mfg_regional_df.to_csv(ecos_dir / 'ecos_industrial_production_index.csv', index=False)
        print(f"제조업생산지수 지역별 데이터 생성: {len(mfg_regional_df)} 건")

if __name__ == "__main__":
    process_ecos_data()
