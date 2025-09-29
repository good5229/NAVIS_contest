#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECOS 데이터 전체 시계열 수집기 (1997-2025년)
"""

import requests
import pandas as pd
import json
import os
from pathlib import Path
import time

def collect_ecos_full_timeseries():
    """ECOS 데이터 1997-2025년 전체 시계열 수집"""
    api_key = os.environ.get('ECOS_API_KEY')
    if not api_key:
        print("ECOS_API_KEY가 설정되지 않았습니다.")
        return
    
    data_dir = Path('data/ecos')
    data_dir.mkdir(exist_ok=True)
    
    print("🚀 ECOS 전체 시계열 데이터 수집 시작 (1997-2025년)")
    
    # 1. 소비자물가지수 수집 (1997-2025년)
    print("\n📊 소비자물가지수 수집 중...")
    cpi_data = []
    
    # 연도별로 수집 (API 한번에 너무 많은 데이터 요청 방지)
    for year in range(1997, 2026):
        start_month = f"{year}01"
        end_month = f"{year}12"
        cpi_url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/100/901Y009/M/{start_month}/{end_month}/0"
        
        try:
            print(f"  {year}년 데이터 수집 중...", end=" ")
            response = requests.get(cpi_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                    rows = data['StatisticSearch']['row']
                    year_data = 0
                    for row in rows:
                        time_str = row['TIME']
                        month = int(time_str[4:6])
                        value = float(row['DATA_VALUE'])
                        
                        cpi_data.append({
                            'region': '전국',
                            'year': year,
                            'month': month,
                            'value': value,
                            'indicator': 'cpi'
                        })
                        year_data += 1
                    print(f"✅ {year_data}건")
                else:
                    print("❌ 데이터 없음")
            else:
                print(f"❌ API 오류: {response.status_code}")
            
            time.sleep(0.5)  # API 호출 간격
            
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    if cpi_data:
        df = pd.DataFrame(cpi_data)
        df.to_csv(data_dir / 'ecos_cpi_full_timeseries.csv', index=False)
        print(f"✅ 소비자물가지수 전체 데이터 저장: {len(df)} 건")
    
    # 2. 제조업생산지수 수집 (1997-2025년)
    print("\n🏭 제조업생산지수 수집 중...")
    mfg_data = []
    
    for year in range(1997, 2026):
        start_month = f"{year}01"
        end_month = f"{year}12"
        mfg_url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/100/901Y034/M/{start_month}/{end_month}/I31AA/I10A"
        
        try:
            print(f"  {year}년 데이터 수집 중...", end=" ")
            response = requests.get(mfg_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                    rows = data['StatisticSearch']['row']
                    year_data = 0
                    for row in rows:
                        time_str = row['TIME']
                        month = int(time_str[4:6])
                        value = float(row['DATA_VALUE'])
                        
                        mfg_data.append({
                            'region': '제조업전체',
                            'year': year,
                            'month': month,
                            'value': value,
                            'indicator': 'manufacturing_production_index'
                        })
                        year_data += 1
                    print(f"✅ {year_data}건")
                else:
                    print("❌ 데이터 없음")
            else:
                print(f"❌ API 오류: {response.status_code}")
            
            time.sleep(0.5)  # API 호출 간격
            
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    if mfg_data:
        df = pd.DataFrame(mfg_data)
        df.to_csv(data_dir / 'ecos_manufacturing_full_timeseries.csv', index=False)
        print(f"✅ 제조업생산지수 전체 데이터 저장: {len(df)} 건")
    
    # 3. 연도별 집계 데이터 생성
    print("\n📈 연도별 집계 데이터 생성 중...")
    
    if cpi_data and mfg_data:
        # 17개 지역 리스트
        regions = [
            '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', 
            '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도', 
            '충청북도', '충청남도', '전라북도', '전라남도', '경상북도', 
            '경상남도', '제주특별자치도'
        ]
        
        # CPI 연도별 평균 계산 후 지역별 복제
        cpi_df = pd.DataFrame(cpi_data)
        cpi_yearly = cpi_df.groupby('year')['value'].mean().reset_index()
        
        cpi_regional = []
        for _, row in cpi_yearly.iterrows():
            for region in regions:
                cpi_regional.append({
                    'region': region,
                    'year': row['year'],
                    'value': row['value']
                })
        
        cpi_regional_df = pd.DataFrame(cpi_regional)
        cpi_regional_df.to_csv(data_dir / 'ecos_cpi_regional_timeseries.csv', index=False)
        print(f"✅ CPI 지역별 시계열 데이터: {len(cpi_regional_df)} 건")
        
        # 제조업생산지수 연도별 평균 계산 후 지역별 복제
        mfg_df = pd.DataFrame(mfg_data)
        mfg_yearly = mfg_df.groupby('year')['value'].mean().reset_index()
        
        mfg_regional = []
        for _, row in mfg_yearly.iterrows():
            for region in regions:
                mfg_regional.append({
                    'region': region,
                    'year': row['year'],
                    'value': row['value']
                })
        
        mfg_regional_df = pd.DataFrame(mfg_regional)
        mfg_regional_df.to_csv(data_dir / 'ecos_manufacturing_regional_timeseries.csv', index=False)
        print(f"✅ 제조업생산지수 지역별 시계열 데이터: {len(mfg_regional_df)} 건")
        
        # 요약 통계
        print("\n📊 수집 완료 요약:")
        print(f"  • 소비자물가지수: {len(cpi_data)} 건 (1997-2025년)")
        print(f"  • 제조업생산지수: {len(mfg_data)} 건 (1997-2025년)")
        print(f"  • 지역별 CPI 데이터: {len(cpi_regional_df)} 건")
        print(f"  • 지역별 제조업 데이터: {len(mfg_regional_df)} 건")
        
        # 연도별 데이터 포인트 확인
        print(f"\n📅 연도별 데이터 포인트:")
        years_with_data = sorted(cpi_yearly['year'].unique())
        print(f"  • 데이터 연도: {years_with_data[0]}-{years_with_data[-1]} ({len(years_with_data)}년)")
        
        return True
    
    return False

if __name__ == "__main__":
    success = collect_ecos_full_timeseries()
    if success:
        print("\n🎉 ECOS 전체 시계열 데이터 수집 완료!")
    else:
        print("\n❌ 데이터 수집 실패")
