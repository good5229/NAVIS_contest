#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NABIS 공모전 객관지표에서 경제 관련 지표 시계열 분석
"""

import pandas as pd
from pathlib import Path
import sys
import traceback

def analyze_economic_indicators():
    """경제 관련 지표들의 시계열 분석"""
    
    print("🚀 NABIS 객관지표 경제 관련 지표 시계열 분석 시작")
    
    # 경제 관련 지표 파일들
    economic_indicators = {
        '9.1_지역내총생산.xlsx': 'GRDP (지역내총생산)',
        '9.2_지역내 무역거래량.xlsx': '무역거래량',
        '9.3_1인당 민간소비지출액.xlsx': '1인당 민간소비지출',
        '10.1_소비자물가상승률.xlsx': '소비자물가상승률',
        '10.2_지가변동률.xlsx': '지가변동률',
        '10.3_재정자립도.xlsx': '재정자립도',
        '8.1_최근 3개년 사업체수 증감률.xlsx': '사업체수 증감률',
        '8.2_최근 3개년 종사자수 증감률.xlsx': '종사자수 증감률',
        '8.3_경제활동참가율.xlsx': '경제활동참가율',
        '8.4_연구원당 연구개발비.xlsx': '연구개발비',
        '8.5_지식기반서비스업 입지계수 3개년 평균(종사자기준).xlsx': '지식기반서비스업',
        '8.5_지식기반제조업 입지계수 3개년 평균(종사자기준).xlsx': '지식기반제조업',
        '8.6_최근 3개년 창업기업수 증감률.xlsx': '창업기업수 증감률'
    }
    
    base_path = Path('civil_data/NBABIS 공모전 데이터/(NABIS_공모전)객관지표/(NABIS_공모전)객관지표')
    
    results = []
    
    for filename, description in economic_indicators.items():
        file_path = base_path / filename
        
        if not file_path.exists():
            print(f"❌ 파일 없음: {filename}")
            continue
            
        print(f"\\n📊 분석 중: {description} ({filename})")
        
        try:
            # 엑셀 파일 읽기 (첫 번째 시트)
            df = pd.read_excel(file_path, sheet_name=0)
            
            # 데이터 구조 분석
            print(f"   📋 데이터 크기: {df.shape}")
            print(f"   📋 컬럼: {list(df.columns)[:5]}...")  # 처음 5개만 표시
            
            # 연도 컬럼 찾기
            year_columns = []
            for col in df.columns:
                if isinstance(col, (int, float)) and 2000 <= col <= 2030:
                    year_columns.append(int(col))
                elif isinstance(col, str):
                    # 문자열에서 연도 추출 시도
                    try:
                        year = int(col.replace('년', '').replace('Y', '').strip())
                        if 2000 <= year <= 2030:
                            year_columns.append(year)
                    except:
                        pass
            
            year_columns = sorted(year_columns)
            
            if year_columns:
                start_year = min(year_columns)
                end_year = max(year_columns)
                year_count = len(year_columns)
                
                print(f"   ✅ 시계열 기간: {start_year}-{end_year}년 ({year_count}년간)")
                
                # 지역 수 확인
                region_col = None
                for col in df.columns:
                    if '지역' in str(col) or '시도' in str(col) or '광역' in str(col):
                        region_col = col
                        break
                
                if region_col is not None:
                    regions = df[region_col].dropna().unique()
                    region_count = len(regions)
                    print(f"   📍 지역 수: {region_count}개")
                    print(f"   📍 주요 지역: {list(regions)[:5]}...")
                else:
                    region_count = "미확인"
                    print(f"   📍 지역 정보를 찾을 수 없습니다.")
                
                results.append({
                    'indicator': description,
                    'filename': filename,
                    'start_year': start_year,
                    'end_year': end_year,
                    'year_count': year_count,
                    'region_count': region_count,
                    'data_shape': df.shape
                })
            else:
                print(f"   ❌ 연도 정보를 찾을 수 없습니다.")
                print(f"   📋 샘플 컬럼: {list(df.columns)[:10]}")
                
        except Exception as e:
            print(f"   ❌ 분석 오류: {str(e)}")
            # traceback.print_exc()
    
    return results

def compare_with_current_data(results):
    """현재 보유 데이터와 비교"""
    
    print(f"\\n\\n🔍 현재 보유 데이터와의 비교")
    print("=" * 80)
    
    # 현재 보유 데이터 시계열
    current_data = {
        'ECOS CPI': {'start': 1997, 'end': 2025, 'years': 29},
        'ECOS 제조업지수': {'start': 1997, 'end': 2025, 'years': 29},
        'KOSIS 재정자립도': {'start': 2001, 'end': 2025, 'years': 25},
        'KOSIS GDP (고정)': {'start': 2023, 'end': 2023, 'years': 1}  # 문제가 되는 부분
    }
    
    print("📊 현재 보유 데이터:")
    for name, info in current_data.items():
        print(f"   • {name}: {info['start']}-{info['end']}년 ({info['years']}년간)")
    
    print(f"\\n📊 NABIS 객관지표 비교:")
    print(f"{'지표명':<25} {'기간':<15} {'년수':<6} {'지역수':<8} {'현재 데이터와 비교'}")
    print("-" * 80)
    
    for result in results:
        indicator = result['indicator'][:23]
        period = f"{result['start_year']}-{result['end_year']}"
        years = result['year_count']
        regions = str(result['region_count'])
        
        # 현재 데이터와 비교
        comparison = ""
        if '지역내총생산' in result['indicator'] or 'GRDP' in result['indicator']:
            comparison = f"vs KOSIS GDP: +{years-1}년 더 많음"
        elif '재정자립도' in result['indicator']:
            current_fiscal = current_data['KOSIS 재정자립도']
            if result['start_year'] <= current_fiscal['start'] and result['end_year'] >= current_fiscal['end']:
                comparison = "기간 일치 또는 더 넓음"
            else:
                comparison = f"기간 차이 있음"
        elif '소비자물가' in result['indicator']:
            current_cpi = current_data['ECOS CPI']
            if result['start_year'] >= current_cpi['start'] and result['end_year'] <= current_cpi['end']:
                comparison = "ECOS 데이터가 더 넓음"
            else:
                comparison = f"기간 비교 필요"
        else:
            comparison = "신규 지표"
        
        print(f"{indicator:<25} {period:<15} {years:<6} {regions:<8} {comparison}")

def main():
    """메인 실행 함수"""
    
    try:
        # 경제 지표 분석
        results = analyze_economic_indicators()
        
        if results:
            # 현재 데이터와 비교
            compare_with_current_data(results)
            
            print(f"\\n✅ 총 {len(results)}개 경제 관련 지표 분석 완료!")
            
            # 주요 발견사항
            print(f"\\n🎯 주요 발견사항:")
            grdp_found = any('지역내총생산' in r['indicator'] for r in results)
            if grdp_found:
                grdp_data = next(r for r in results if '지역내총생산' in r['indicator'])
                print(f"   • GRDP 데이터 발견: {grdp_data['start_year']}-{grdp_data['end_year']}년")
                print(f"   • 현재 GDP 고정값 문제 해결 가능!")
            
            fiscal_found = any('재정자립도' in r['indicator'] for r in results)
            if fiscal_found:
                print(f"   • 재정자립도 데이터 확인됨 (검증 가능)")
            
        else:
            print("❌ 분석 가능한 경제 지표를 찾지 못했습니다.")
            return 1
            
    except Exception as e:
        print(f"❌ 전체 분석 오류: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
