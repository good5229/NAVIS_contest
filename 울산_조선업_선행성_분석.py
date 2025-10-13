#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
울산 조선업과 BDS 선행성 분석
실제 조선업 데이터를 활용한 선행성 검증
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_ulsan_shipbuilding_leading_indicator():
    """울산 조선업과 BDS 선행성 분석"""
    
    print("🚢 울산 조선업과 BDS 선행성 분석 시작")
    print("="*60)
    
    # 1. 실제 데이터 정의
    # 울산 BDS 데이터 (2015-2019)
    ulsan_bds_data = {
        '2015': 4.65,
        '2016': 4.68,
        '2017': 4.71,
        '2018': 4.69,
        '2019': 4.72
    }
    
    # 울산 조선업 생산액 데이터 (10억원 단위)
    ulsan_shipbuilding_data = {
        '2015': 37769,
        '2016': 38167,
        '2017': 37825,
        '2018': 36394,
        '2019': 37496,
        '2020': 33220,
        '2021': 39122,
        '2022': 48328,
        '2023': 40058
    }
    
    # 2. 데이터 전처리
    years = list(map(int, ulsan_bds_data.keys()))
    bds_scores = list(ulsan_bds_data.values())
    shipbuilding_values = [ulsan_shipbuilding_data[str(year)] for year in years]
    
    print(f"📊 분석 기간: {years[0]}-{years[-1]}")
    print(f"📈 BDS 점수: {bds_scores[0]:.2f} → {bds_scores[-1]:.2f}")
    print(f"🚢 조선업 생산액: {shipbuilding_values[0]:,} → {shipbuilding_values[-1]:,} (10억원)")
    
    # 3. 상관관계 분석
    correlation = np.corrcoef(bds_scores, shipbuilding_values)[0,1]
    print(f"\n🔗 BDS-조선업 상관계수: {correlation:.3f}")
    
    # 4. 선행성 분석 (BDS가 조선업보다 1년 선행)
    bds_leading = bds_scores[:-1]  # BDS 2015-2018
    shipbuilding_lagged = shipbuilding_values[1:]  # 조선업 2016-2019
    
    leading_correlation = np.corrcoef(bds_leading, shipbuilding_lagged)[0,1]
    print(f"⏰ BDS 선행성 상관계수 (1년 선행): {leading_correlation:.3f}")
    
    # 5. 성장률 분석
    bds_growth_rates = [(bds_scores[i] - bds_scores[i-1]) / bds_scores[i-1] * 100 
                        for i in range(1, len(bds_scores))]
    shipbuilding_growth_rates = [(shipbuilding_values[i] - shipbuilding_values[i-1]) / shipbuilding_values[i-1] * 100 
                                for i in range(1, len(shipbuilding_values))]
    
    print(f"\n📈 BDS 연평균 성장률: {np.mean(bds_growth_rates):.2f}%")
    print(f"📈 조선업 연평균 성장률: {np.mean(shipbuilding_growth_rates):.2f}%")
    
    # 6. 선행성 검증 (방향성 일치도)
    direction_matches = 0
    for i in range(len(bds_growth_rates)):
        if (bds_growth_rates[i] > 0 and shipbuilding_growth_rates[i+1] > 0) or \
           (bds_growth_rates[i] < 0 and shipbuilding_growth_rates[i+1] < 0):
            direction_matches += 1
    
    direction_accuracy = direction_matches / len(bds_growth_rates) * 100
    print(f"🎯 방향성 일치도: {direction_accuracy:.1f}%")
    
    # 7. 결과 저장
    analysis_results = {
        "analysis_period": f"{years[0]}-{years[-1]}",
        "region": "울산광역시",
        "data_sources": {
            "bds": "실제 BDS 데이터 (2015-2019)",
            "shipbuilding": "실제 조선업 생산액 데이터 (10억원 단위)"
        },
        "correlation_analysis": {
            "bds_shipbuilding_correlation": float(correlation),
            "bds_leading_correlation": float(leading_correlation),
            "direction_accuracy": float(direction_accuracy)
        },
        "growth_analysis": {
            "bds_avg_growth": float(np.mean(bds_growth_rates)),
            "shipbuilding_avg_growth": float(np.mean(shipbuilding_growth_rates))
        },
        "leading_indicator_validation": {
            "bds_leads_shipbuilding": leading_correlation > correlation,
            "statistical_significance": abs(leading_correlation) > 0.7,
            "directional_consistency": direction_accuracy > 60
        },
        "key_findings": [
            f"BDS-조선업 상관계수: {correlation:.3f}",
            f"BDS 1년 선행 상관계수: {leading_correlation:.3f}",
            f"방향성 일치도: {direction_accuracy:.1f}%",
            f"BDS가 조선업보다 선행성을 보임: {leading_correlation > correlation}"
        ]
    }
    
    # JSON 결과 저장
    with open('울산_조선업_선행성_분석_결과.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 분석 결과 저장: 울산_조선업_선행성_분석_결과.json")
    
    return analysis_results, years, bds_scores, shipbuilding_values

def create_shipbuilding_leading_analysis_charts(analysis_results: Dict, years: List, bds_scores: List, shipbuilding_values: List):
    """조선업 선행성 분석 차트 생성"""
    
    print("\n📊 조선업 선행성 분석 차트 생성 중...")
    
    # 1. 메인 분석 차트
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('울산 조선업과 BDS 선행성 분석 (실제 데이터)', fontsize=16, fontweight='bold')
    
    # 1-1. BDS vs 조선업 시계열
    ax1.plot(years, bds_scores, 'o-', linewidth=3, markersize=8, color='#2E86AB', label='BDS')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(years, shipbuilding_values, 's-', linewidth=3, markersize=8, color='#A23B72', label='조선업 생산액')
    
    ax1.set_title('BDS vs 조선업 생산액 시계열', fontsize=14, fontweight='bold')
    ax1.set_xlabel('연도')
    ax1.set_ylabel('BDS 점수', color='#2E86AB')
    ax1_twin.set_ylabel('조선업 생산액 (10억원)', color='#A23B72')
    ax1.grid(True, alpha=0.3)
    
    # 상관계수 표시
    correlation = analysis_results['correlation_analysis']['bds_shipbuilding_correlation']
    ax1.text(0.05, 0.95, f'상관계수: {correlation:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    # 1-2. 선행성 분석 (BDS 1년 선행)
    bds_leading = bds_scores[:-1]
    shipbuilding_lagged = shipbuilding_values[1:]
    leading_years = years[:-1]
    
    ax2.plot(leading_years, bds_leading, 'o-', linewidth=3, markersize=8, color='#2E86AB', label='BDS (t)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(leading_years, shipbuilding_lagged, 's-', linewidth=3, markersize=8, color='#A23B72', label='조선업 (t+1)')
    
    ax2.set_title('BDS 선행성 분석 (1년 선행)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('연도')
    ax2.set_ylabel('BDS 점수', color='#2E86AB')
    ax2_twin.set_ylabel('조선업 생산액 (10억원)', color='#A23B72')
    ax2.grid(True, alpha=0.3)
    
    # 선행성 상관계수 표시
    leading_correlation = analysis_results['correlation_analysis']['bds_leading_correlation']
    ax2.text(0.05, 0.95, f'선행성 상관계수: {leading_correlation:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    # 1-3. 성장률 비교
    bds_growth = [(bds_scores[i] - bds_scores[i-1]) / bds_scores[i-1] * 100 for i in range(1, len(bds_scores))]
    shipbuilding_growth = [(shipbuilding_values[i] - shipbuilding_values[i-1]) / shipbuilding_values[i-1] * 100 for i in range(1, len(shipbuilding_values))]
    growth_years = years[1:]
    
    ax3.plot(growth_years, bds_growth, 'o-', linewidth=3, markersize=8, color='#2E86AB', label='BDS 성장률')
    ax3.plot(growth_years, shipbuilding_growth, 's-', linewidth=3, markersize=8, color='#A23B72', label='조선업 성장률')
    ax3.set_title('연도별 성장률 비교', fontsize=14, fontweight='bold')
    ax3.set_xlabel('연도')
    ax3.set_ylabel('성장률 (%)')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 1-4. 선행성 검증 결과
    validation_results = analysis_results['leading_indicator_validation']
    categories = ['BDS 선행성', '통계적 유의성', '방향성 일치']
    values = [
        validation_results['bds_leads_shipbuilding'],
        validation_results['statistical_significance'],
        validation_results['directional_consistency']
    ]
    colors = ['#28a745' if v else '#dc3545' for v in values]
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_title('선행성 검증 결과', fontsize=14, fontweight='bold')
    ax4.set_ylabel('검증 통과 여부')
    ax4.set_ylim(0, 1.2)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 결과 표시
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                '통과' if value else '실패', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('울산_조선업_선행성_분석_차트.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 조선업 선행성 분석 차트 생성 완료!")
    print("📁 생성된 파일: 울산_조선업_선행성_분석_차트.png")

def main():
    """메인 실행 함수"""
    print("🚀 울산 조선업과 BDS 선행성 분석 시작")
    print("="*60)
    
    # 1. 선행성 분석 실행
    analysis_results, years, bds_scores, shipbuilding_values = analyze_ulsan_shipbuilding_leading_indicator()
    
    # 2. 차트 생성
    create_shipbuilding_leading_analysis_charts(analysis_results, years, bds_scores, shipbuilding_values)
    
    # 3. 결과 요약
    print("\n📋 울산 조선업 선행성 분석 요약")
    print("="*60)
    print(f"🎯 분석 기간: {analysis_results['analysis_period']}")
    print(f"🔗 BDS-조선업 상관계수: {analysis_results['correlation_analysis']['bds_shipbuilding_correlation']:.3f}")
    print(f"⏰ BDS 선행성 상관계수: {analysis_results['correlation_analysis']['bds_leading_correlation']:.3f}")
    print(f"🎯 방향성 일치도: {analysis_results['correlation_analysis']['direction_accuracy']:.1f}%")
    
    print("\n💡 주요 발견사항:")
    for finding in analysis_results['key_findings']:
        print(f"  • {finding}")
    
    print("\n✅ 울산 조선업 선행성 분석 완료!")
    print("📁 생성된 파일:")
    print("  • 울산_조선업_선행성_분석_결과.json")
    print("  • 울산_조선업_선행성_분석_차트.png")

if __name__ == "__main__":
    main()
