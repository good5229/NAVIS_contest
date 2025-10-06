#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KOSIS GRDP 데이터 분석 및 EU RCI 방식 가중치 계산
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def analyze_kosis_grdp_data():
    """KOSIS GRDP 데이터 시계열 분석"""
    
    print("🚀 KOSIS GRDP 데이터 분석 시작")
    print("="*60)
    
    # 파일 경로
    grdp_file = Path('data/kosis/시도별 GRDP.xlsx')
    per_capita_file = Path('data/kosis/시도별 1인당 GRDP.xlsx')
    
    results = {}
    
    try:
        # 시도별 GRDP 데이터 읽기
        print("📊 시도별 GRDP 데이터 분석...")
        grdp_df = pd.read_excel(grdp_file, sheet_name=0)
        print(f"   데이터 크기: {grdp_df.shape}")
        print(f"   컬럼: {list(grdp_df.columns)}")
        
        # 1인당 GRDP 데이터 읽기
        print("\\n📊 시도별 1인당 GRDP 데이터 분석...")
        per_capita_df = pd.read_excel(per_capita_file, sheet_name=0)
        print(f"   데이터 크기: {per_capita_df.shape}")
        print(f"   컬럼: {list(per_capita_df.columns)}")
        
        # 연도 컬럼 찾기
        year_columns = []
        for col in grdp_df.columns:
            if isinstance(col, (int, float)) and 2010 <= col <= 2030:
                year_columns.append(int(col))
        
        year_columns = sorted(year_columns)
        print(f"\\n📅 확인된 연도: {year_columns}")
        print(f"📅 시계열 기간: {min(year_columns)}-{max(year_columns)}년 ({len(year_columns)}년간)")
        
        results['years'] = year_columns
        results['grdp_data'] = grdp_df
        results['per_capita_data'] = per_capita_df
        
        return results
        
    except Exception as e:
        print(f"❌ 데이터 분석 오류: {str(e)}")
        return None

def calculate_eu_rci_weights(per_capita_grdp: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """EU RCI 방식으로 지역별 가중치 계산"""
    
    print("\\n🎯 EU RCI 방식 가중치 계산")
    print("="*60)
    
    # 1인당 GRDP를 달러로 변환 (1달러 = 1300원 가정)
    usd_conversion_rate = 1300
    
    # 발전단계 분류 기준 (달러)
    stage_thresholds = {
        'factor_driven': 17000,      # 요소주도 < $17,000
        'efficiency_driven': 25000   # 효율성주도 < $25,000, 혁신주도 >= $25,000
    }
    
    # 가중치 방안들
    weight_schemes = {
        'scheme_1_basic': {
            'description': 'EU RCI 기본 방식 (3단계)',
            'factor_driven': {'basic': 0.5, 'efficiency': 0.3, 'innovation': 0.2},
            'efficiency_driven': {'basic': 0.3, 'efficiency': 0.5, 'innovation': 0.2}, 
            'innovation_driven': {'basic': 0.2, 'efficiency': 0.3, 'innovation': 0.5}
        },
        'scheme_2_gradual': {
            'description': '점진적 가중치 변화 (연속함수)',
            'formula': 'GDP 수준에 따른 연속적 가중치 조정'
        },
        'scheme_3_korean': {
            'description': '한국형 지역발전 가중치 (균형발전 중심)',
            'high_income': {'grdp': 0.25, 'fiscal': 0.25, 'manufacturing': 0.25, 'service': 0.25},
            'middle_income': {'grdp': 0.35, 'fiscal': 0.30, 'manufacturing': 0.20, 'service': 0.15},
            'low_income': {'grdp': 0.40, 'fiscal': 0.35, 'manufacturing': 0.15, 'service': 0.10}
        }
    }
    
    # 지역별 발전단계 분류
    region_stages = {}
    region_weights = {}
    
    for region, per_capita in per_capita_grdp.items():
        # 원화를 달러로 변환 (천원 단위이므로 * 1000 / 1300)
        per_capita_usd = (per_capita * 1000) / usd_conversion_rate
        
        if per_capita_usd < stage_thresholds['factor_driven']:
            stage = 'factor_driven'
            stage_name = '요소주도형'
        elif per_capita_usd < stage_thresholds['efficiency_driven']:
            stage = 'efficiency_driven' 
            stage_name = '효율성주도형'
        else:
            stage = 'innovation_driven'
            stage_name = '혁신주도형'
        
        region_stages[region] = {
            'stage': stage,
            'stage_name': stage_name,
            'per_capita_krw': per_capita,
            'per_capita_usd': int(per_capita_usd)
        }
    
    # 각 방안별 가중치 계산
    for scheme_name, scheme_info in weight_schemes.items():
        print(f"\\n🔧 {scheme_name}: {scheme_info['description']}")
        scheme_weights = {}
        
        for region, stage_info in region_stages.items():
            stage = stage_info['stage']
            
            if scheme_name == 'scheme_1_basic':
                weights = scheme_info[stage]
            elif scheme_name == 'scheme_2_gradual':
                # 연속함수 방식 (GDP 수준에 비례한 가중치)
                per_capita_usd = stage_info['per_capita_usd']
                innovation_weight = min(0.5, max(0.2, (per_capita_usd - 15000) / 20000))
                basic_weight = 0.5 - innovation_weight * 0.3
                efficiency_weight = 1.0 - basic_weight - innovation_weight
                
                weights = {
                    'basic': round(basic_weight, 3),
                    'efficiency': round(efficiency_weight, 3), 
                    'innovation': round(innovation_weight, 3)
                }
            elif scheme_name == 'scheme_3_korean':
                # 한국형 BDS 가중치
                if stage_info['per_capita_usd'] >= 30000:
                    weights = scheme_info['high_income']
                elif stage_info['per_capita_usd'] >= 20000:
                    weights = scheme_info['middle_income']
                else:
                    weights = scheme_info['low_income']
            
            scheme_weights[region] = weights
            
            print(f"   {region:<12} ({stage_info['stage_name']:<8}, ${stage_info['per_capita_usd']:,}): {weights}")
        
        region_weights[scheme_name] = scheme_weights
    
    return region_weights, region_stages

def recommend_best_scheme(region_weights: Dict, region_stages: Dict) -> str:
    """가장 합리적인 가중치 방안 추천"""
    
    print("\\n🎯 가중치 방안 평가 및 추천")
    print("="*60)
    
    # 각 방안의 장단점 분석
    evaluations = {
        'scheme_1_basic': {
            'pros': ['EU 공식 방법론', '이론적 근거 명확', '국제 비교 가능'],
            'cons': ['한국 상황 미반영', '급격한 가중치 변화', '3단계 분류의 한계'],
            'score': 7
        },
        'scheme_2_gradual': {
            'pros': ['연속적 변화', '세밀한 조정', '급격한 변화 방지'],
            'cons': ['복잡한 계산', '해석 어려움', '이론적 근거 부족'],
            'score': 6
        },
        'scheme_3_korean': {
            'pros': ['한국 현실 반영', 'BDS 지표 특화', '균형발전 목표 부합'],
            'cons': ['경험적 설정', '국제 비교 제한', '검증 필요'],
            'score': 8
        }
    }
    
    print("📊 방안별 평가:")
    for scheme, eval_info in evaluations.items():
        print(f"\\n{scheme}:")
        print(f"   장점: {', '.join(eval_info['pros'])}")
        print(f"   단점: {', '.join(eval_info['cons'])}")
        print(f"   점수: {eval_info['score']}/10")
    
    # 추천 방안
    best_scheme = max(evaluations.keys(), key=lambda x: evaluations[x]['score'])
    
    print(f"\\n🏆 추천 방안: {best_scheme}")
    print(f"   이유: 한국의 지역균형발전 정책 목표에 가장 적합하며, BDS 지표의 특성을 잘 반영")
    
    return best_scheme

def apply_weights_to_bds(best_scheme: str, region_weights: Dict) -> Dict:
    """추천 가중치를 적용한 BDS 계산 예시"""
    
    print(f"\\n🔧 {best_scheme} 가중치 적용 BDS 계산")
    print("="*60)
    
    # 샘플 데이터 (실제로는 KOSIS/ECOS 데이터 사용)
    sample_data = {
        '서울특별시': {'grdp': 1.0, 'fiscal': 0.8, 'manufacturing': 0.6, 'service': 1.0},
        '경기도': {'grdp': 0.9, 'fiscal': 0.7, 'manufacturing': 0.8, 'service': 0.9},
        '부산광역시': {'grdp': 0.6, 'fiscal': 0.5, 'manufacturing': 0.7, 'service': 0.6},
        '전라북도': {'grdp': 0.3, 'fiscal': 0.3, 'manufacturing': 0.4, 'service': 0.3}
    }
    
    bds_results = {}
    
    print("📊 지역별 가중치 적용 BDS 계산:")
    for region in sample_data.keys():
        if region in region_weights[best_scheme]:
            weights = region_weights[best_scheme][region]
            indicators = sample_data[region]
            
            # BDS 계산
            bds_score = (
                indicators['grdp'] * weights['grdp'] +
                indicators['fiscal'] * weights['fiscal'] +
                indicators['manufacturing'] * weights['manufacturing'] +
                indicators['service'] * weights['service']
            ) * 10
            
            bds_results[region] = {
                'bds_score': round(bds_score, 2),
                'weights': weights,
                'indicators': indicators
            }
            
            print(f"   {region:<12}: BDS {bds_score:.2f} (가중치: GRDP {weights['grdp']}, 재정 {weights['fiscal']}, 제조업 {weights['manufacturing']}, 서비스업 {weights['service']})")
    
    return bds_results

def main():
    """메인 실행 함수"""
    
    # 1. KOSIS 데이터 분석
    data_results = analyze_kosis_grdp_data()
    
    if data_results is None:
        print("❌ 데이터 분석 실패")
        return 1
    
    # 2. 샘플 1인당 GRDP 데이터 (실제로는 엑셀에서 추출)
    sample_per_capita_grdp = {
        '서울특별시': 45000,  # 천원
        '경기도': 35000,
        '인천광역시': 32000,
        '부산광역시': 28000,
        '대구광역시': 26000,
        '대전광역시': 30000,
        '광주광역시': 25000,
        '울산광역시': 55000,
        '세종특별자치시': 40000,
        '강원도': 22000,
        '충청북도': 28000,
        '충청남도': 35000,
        '전라북도': 20000,
        '전라남도': 25000,
        '경상북도': 26000,
        '경상남도': 30000,
        '제주특별자치도': 24000
    }
    
    # 3. EU RCI 방식 가중치 계산
    region_weights, region_stages = calculate_eu_rci_weights(sample_per_capita_grdp)
    
    # 4. 최적 방안 추천
    best_scheme = recommend_best_scheme(region_weights, region_stages)
    
    # 5. 추천 가중치 적용 BDS 계산
    bds_results = apply_weights_to_bds(best_scheme, region_weights)
    
    print("\\n✅ 분석 완료!")
    print("\\n🎯 다음 단계:")
    print("   1. 실제 KOSIS 1인당 GRDP 데이터 추출")
    print("   2. 추천 가중치 방안으로 전체 BDS 재계산")
    print("   3. 기존 BDS와 비교 분석")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
