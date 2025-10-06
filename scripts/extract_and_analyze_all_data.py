#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAVIS GRDP 데이터 추출 및 BDS 계산 개선 스크립트
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

def extract_navis_grdp() -> Optional[pd.DataFrame]:
    """NAVIS 엑셀 파일에서 GRDP 데이터 추출"""
    
    print("🚀 NAVIS GRDP 데이터 추출 시작")
    
    navis_file = Path('civil_data/NBABIS 공모전 데이터/(NABIS_공모전)객관지표/(NABIS_공모전)객관지표/9.1_지역내총생산.xlsx')
    
    if not navis_file.exists():
        print(f"❌ 파일이 없습니다: {navis_file}")
        return None
    
    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(navis_file, sheet_name=0)
        print(f"📊 원본 데이터 크기: {df.shape}")
        
        # 데이터 구조 분석
        print("📋 컬럼 정보:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        # 시도별 데이터만 필터링 (시군구 제외)
        if '시도' in df.columns:
            sido_data = df[df['시군구'] == '전체'].copy()
        else:
            # 컬럼명이 다를 수 있으므로 광역시도만 필터링
            sido_data = df[df.iloc[:, 6] == '전체'].copy() if df.shape[1] > 6 else df.copy()
        
        print(f"📊 시도별 데이터 크기: {sido_data.shape}")
        
        # 연도별 데이터 추출 (2019-2024년)
        grdp_data = []
        
        for _, row in sido_data.iterrows():
            region = row.iloc[5] if df.shape[1] > 5 else row.iloc[0]  # 시도명
            
            # 연도별 값 추출 (컬럼명에서 연도 찾기)
            for col in df.columns:
                if '2019' in str(col) or '2020' in str(col) or '2021' in str(col) or \
                   '2022' in str(col) or '2023' in str(col) or '2024' in str(col):
                    year = int(''.join(filter(str.isdigit, str(col)))[:4])
                    value = row[col]
                    
                    if pd.notna(value) and value != 0:
                        grdp_data.append({
                            'year': year,
                            'region': region,
                            'grdp': float(value)
                        })
        
        if grdp_data:
            result_df = pd.DataFrame(grdp_data)
            print(f"✅ 추출된 GRDP 데이터: {len(result_df)}건")
            
            # 연도별 지역 수 확인
            year_summary = result_df.groupby('year')['region'].count()
            print("📊 연도별 지역 수:")
            for year, count in year_summary.items():
                print(f"  {year}년: {count}개 지역")
            
            return result_df
        else:
            print("❌ GRDP 데이터를 추출할 수 없습니다.")
            return None
            
    except Exception as e:
        print(f"❌ GRDP 추출 오류: {str(e)}")
        return None

def create_comprehensive_analysis():
    """종합 데이터 분석 보고서 생성"""
    
    print("\n" + "="*80)
    print("📊 경제 지표 데이터 종합 분석 보고서")
    print("="*80)
    
    # 현재 보유 데이터
    current_data = {
        'ECOS CPI (전국)': {'start': 1997, 'end': 2025, 'years': 29, 'regions': 1, 'status': '✅ 우수'},
        'ECOS 제조업지수 (전국)': {'start': 1997, 'end': 2025, 'years': 29, 'regions': 1, 'status': '✅ 우수'},
        'KOSIS 재정자립도': {'start': 2001, 'end': 2025, 'years': 25, 'regions': 17, 'status': '✅ 우수'},
        'KOSIS GRDP (고정)': {'start': 2023, 'end': 2023, 'years': 1, 'regions': 17, 'status': '❌ 치명적 결함'}
    }
    
    # 새로 발견된 데이터
    new_data = {
        'NAVIS GRDP': {'start': 2019, 'end': 2024, 'years': 6, 'regions': 17, 'status': '🎯 핵심 해결책'},
        'NAVIS 재정자립도': {'start': 2021, 'end': 2024, 'years': 4, 'regions': 17, 'status': '✅ 검증 가능'},
        'NAVIS 소비자물가': {'start': 2021, 'end': 2024, 'years': 4, 'regions': 17, 'status': '⚡ 지역별 차별화'},
        'KOSIS 제조업지수 (시도별)': {'start': 2015, 'end': 2024, 'years': 10, 'regions': 17, 'status': '🚀 대폭 개선'},
        'KOSIS 서비스업지수 (시도별)': {'start': 2020, 'end': 2025, 'years': 6, 'regions': 17, 'status': '🆕 신규 지표'}
    }
    
    print("\n🔍 현재 보유 데이터:")
    print(f"{'지표명':<25} {'기간':<15} {'년수':<6} {'지역':<6} {'상태'}")
    print("-" * 70)
    for name, info in current_data.items():
        period = f"{info['start']}-{info['end']}"
        print(f"{name:<25} {period:<15} {info['years']:<6} {info['regions']:<6} {info['status']}")
    
    print("\n🆕 새로 발견된 데이터:")
    print(f"{'지표명':<25} {'기간':<15} {'년수':<6} {'지역':<6} {'상태'}")
    print("-" * 70)
    for name, info in new_data.items():
        period = f"{info['start']}-{info['end']}"
        print(f"{name:<25} {period:<15} {info['years']:<6} {info['regions']:<6} {info['status']}")
    
    print("\n🎯 핵심 문제 및 해결 방안:")
    print("1. ❌ 현재 문제: GRDP 2023년 고정값 → 서울시 등 BDS 동일값 문제")
    print("2. 🎯 해결책: NAVIS GRDP 2019-2024년 데이터 활용")
    print("3. 🚀 추가 개선: KOSIS 시도별 제조업/서비스업 지수 활용")
    
    print("\n📈 개선 후 예상 BDS 모델:")
    print("   • GRDP (30%): 2019-2024년 실제 연도별 데이터")
    print("   • 재정자립도 (25%): 2001-2025년 (기존 유지)")
    print("   • 제조업지수 (25%): 시도별 2015-2024년 (전국→지역별 개선)")
    print("   • 서비스업지수 (20%): 시도별 2020-2025년 (신규 추가)")
    
    print("\n🎉 예상 효과:")
    print("   ✅ BDS 신뢰성 대폭 향상 (동일값 문제 해결)")
    print("   ✅ 지역별 특성 반영 (제조업/서비스업 지역별 차이)")
    print("   ✅ 실제 경제 변동 반영 (연도별 GRDP 변화)")
    print("   ✅ 시계열 분석 가능 (2019-2024년 6년간 추이)")

def main():
    """메인 실행 함수"""
    
    # NAVIS GRDP 데이터 추출 시도
    grdp_df = extract_navis_grdp()
    
    if grdp_df is not None:
        # 샘플 데이터 출력
        print("\n📋 NAVIS GRDP 샘플 데이터:")
        print(grdp_df.head(10))
        
        # CSV 저장
        output_file = Path('data/navis/navis_grdp_timeseries.csv')
        output_file.parent.mkdir(exist_ok=True)
        grdp_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ GRDP 데이터 저장: {output_file}")
    
    # 종합 분석 보고서
    create_comprehensive_analysis()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
