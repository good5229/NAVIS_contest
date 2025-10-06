#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 ECOS 데이터 수집기
"""

import requests
import pandas as pd
import json
import os
from pathlib import Path

def collect_ecos_data():
    """ECOS 데이터 수집"""
    api_key = os.environ.get('ECOS_API_KEY')
    if not api_key:
        print("ECOS_API_KEY가 설정되지 않았습니다.")
        return
    
    data_dir = Path('data/ecos')
    data_dir.mkdir(exist_ok=True)
    
    # 1. 소비자물가지수 수집
    print("소비자물가지수 수집 중...")
    cpi_url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/100/901Y009/M/202001/202412/0"
    
    try:
        response = requests.get(cpi_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                rows = data['StatisticSearch']['row']
                cpi_data = []
                
                for row in rows:
                    time_str = row['TIME']
                    year = int(time_str[:4])
                    month = int(time_str[4:6])
                    value = float(row['DATA_VALUE'])
                    
                    cpi_data.append({
                        'region': '전국',
                        'year': year,
                        'month': month,
                        'value': value,
                        'indicator': 'cpi'
                    })
                
                df = pd.DataFrame(cpi_data)
                df.to_csv(data_dir / 'ecos_cpi_national.csv', index=False)
                print(f"소비자물가지수 데이터 저장 완료: {len(df)} 건")
            else:
                print("소비자물가지수 데이터가 없습니다.")
        else:
            print(f"소비자물가지수 API 오류: {response.status_code}")
    except Exception as e:
        print(f"소비자물가지수 수집 오류: {e}")
    
    # 2. 제조업생산지수 수집 (자본재 - 생산지수 원지수)
    print("제조업생산지수 수집 중...")
    mfg_url = f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/100/901Y034/M/202001/202412/I31AA/I10A"
    
    try:
        response = requests.get(mfg_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                rows = data['StatisticSearch']['row']
                mfg_data = []
                
                for row in rows:
                    time_str = row['TIME']
                    year = int(time_str[:4])
                    month = int(time_str[4:6])
                    value = float(row['DATA_VALUE'])
                    
                    mfg_data.append({
                        'region': '제조업전체',
                        'year': year,
                        'month': month,
                        'value': value,
                        'indicator': 'manufacturing_production_index'
                    })
                
                df = pd.DataFrame(mfg_data)
                df.to_csv(data_dir / 'ecos_manufacturing_production_index.csv', index=False)
                print(f"제조업생산지수 데이터 저장 완료: {len(df)} 건")
            else:
                print("제조업생산지수 데이터가 없습니다.")
        else:
            print(f"제조업생산지수 API 오류: {response.status_code}")
    except Exception as e:
        print(f"제조업생산지수 수집 오류: {e}")

if __name__ == "__main__":
    collect_ecos_data()
