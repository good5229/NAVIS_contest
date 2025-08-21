#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시뮬레이션 파라미터 기반 실행기
- 명령행 파라미터를 받아서 시뮬레이션 실행
- 결과를 파라미터별 폴더에 저장
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# navis_analysis 모듈에서 필요한 함수들 import
from navis_analysis import (
    load_and_clean_navis_data,
    compare_bds_with_navis,
    analyze_regional_balance_indicators,
    create_realistic_simulation,
    analyze_investment_priority,
    validate_improvement_potential,
    calculate_investment_impact,
    visualize_simulation,
    create_simulation_dashboard,
    create_investment_priority_visualization,
    create_validation_visualization,
    create_summary_visualization,
    visualize_comparison,
    generate_analysis_report
)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='BDS 시뮬레이션 실행기')
    
    # 시뮬레이션 파라미터
    parser.add_argument('--improvement_rate', type=float, default=0.05,
                        help='기본 개선률 (기본값: 0.05)')
    parser.add_argument('--potential_multiplier', type=float, default=0.03,
                        help='개선 잠재력 배수 (기본값: 0.03)')
    parser.add_argument('--diminishing_factor', type=float, default=0.2,
                        help='체감 효과 계수 (기본값: 0.2)')
    parser.add_argument('--investment_budget', type=float, default=1000,
                        help='투자 예산 (억원, 기본값: 1000)')
    parser.add_argument('--simulation_years', type=int, default=5,
                        help='시뮬레이션 연수 (기본값: 5)')
    
    # 출력 관련
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='출력 폴더 접두사 (예: scenario_01)')
    parser.add_argument('--description', type=str, default='',
                        help='시나리오 설명')
    
    # 선택적 파라미터
    parser.add_argument('--target_regions', type=str, nargs='*',
                        help='대상 지역 (기본값: BDS 하위 5개 지역)')
    parser.add_argument('--skip_validation', action='store_true',
                        help='검증 과정 건너뛰기')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='시각화 건너뛰기')
    
    return parser.parse_args()

def create_output_directory(output_prefix):
    """출력 디렉토리 생성"""
    base_dir = Path("outputs_timeseries")
    output_dir = base_dir / output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_parameters(args, output_dir):
    """파라미터를 JSON 파일로 저장"""
    params = {
        'improvement_rate': args.improvement_rate,
        'potential_multiplier': args.potential_multiplier,
        'diminishing_factor': args.diminishing_factor,
        'investment_budget': args.investment_budget,
        'simulation_years': args.simulation_years,
        'target_regions': args.target_regions,
        'description': args.description,
        'timestamp': datetime.now().isoformat(),
        'skip_validation': args.skip_validation,
        'skip_visualization': args.skip_visualization
    }
    
    params_file = output_dir / "simulation_parameters.json"
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    return params

def create_custom_simulation(bds_data, target_regions, args):
    """파라미터를 적용한 커스텀 시뮬레이션 생성"""
    print(f"\n=== 커스텀 시뮬레이션 (개선률: {args.improvement_rate:.3f}) ===")
    
    # 최신 연도 데이터
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    
    # 시뮬레이션 데이터 생성
    simulation_data = []
    
    for year in range(latest_year + 1, latest_year + args.simulation_years + 1):
        year_data = latest_data.copy()
        year_data['period'] = year
        
        # 대상 지역들의 BDS 개선
        for region in target_regions:
            mask = year_data['region_name'] == region
            if mask.any():
                region_data = year_data[mask].iloc[0]
                current_bds = region_data['BDS']
                
                # 개선 잠재력 기반 조정
                from navis_analysis import calculate_single_improvement_potential
                improvement_potential = calculate_single_improvement_potential(region_data)
                potential_adjustment = improvement_potential * args.potential_multiplier
                
                # 현재 BDS 수준에 따른 조정
                bds_level_adjustment = max(0, (0.5 - abs(current_bds)) / 0.5) * 0.02
                
                # 최종 개선률
                total_improvement_rate = args.improvement_rate + potential_adjustment + bds_level_adjustment
                
                # 체감 효과
                years_passed = year - latest_year
                diminishing_factor = 1 / (1 + years_passed * args.diminishing_factor)
                final_improvement_rate = total_improvement_rate * diminishing_factor
                
                # BDS 개선 적용
                improvement_factor = 1 + final_improvement_rate
                year_data.loc[mask, 'BDS'] *= improvement_factor
        
        simulation_data.append(year_data)
    
    # 결과 결합
    result = pd.concat([bds_data, pd.concat(simulation_data, ignore_index=True)], 
                      ignore_index=True)
    
    return result

def run_simulation(args):
    """전체 시뮬레이션 실행"""
    print(f"=== 시뮬레이션 실행: {args.output_prefix} ===")
    
    # 출력 디렉토리 생성
    output_dir = create_output_directory(args.output_prefix)
    print(f"출력 디렉토리: {output_dir}")
    
    # 파라미터 저장
    params = save_parameters(args, output_dir)
    print(f"파라미터 저장 완료: {params}")
    
    try:
        # 1. 데이터 로드
        print("\n1. 데이터 로드 중...")
        bds_data = pd.read_csv("outputs_timeseries/bds_timeseries.csv")
        print(f"BDS 데이터 로드 완료: {bds_data.shape}")
        
        # NAVIS 데이터 로드 (선택적)
        navis_data = {}
        try:
            navis_data = load_and_clean_navis_data()
            print(f"NAVIS 데이터 로드 완료: {list(navis_data.keys())}")
        except Exception as e:
            print(f"NAVIS 데이터 로드 실패: {e}")
        
        # 2. 대상 지역 선정
        if args.target_regions:
            target_regions = args.target_regions
        else:
            latest_year = bds_data['period'].max()
            latest_data = bds_data[bds_data['period'] == latest_year].copy()
            target_regions = latest_data.nsmallest(5, 'BDS')['region_name'].tolist()
        
        print(f"대상 지역: {target_regions}")
        
        # 3. 시뮬레이션 실행
        print("\n2. 시뮬레이션 실행 중...")
        simulation_data = create_custom_simulation(bds_data, target_regions, args)
        
        # 시뮬레이션 데이터 저장
        simulation_file = output_dir / "simulation_data.csv"
        simulation_data.to_csv(simulation_file, index=False, encoding='utf-8')
        print(f"시뮬레이션 데이터 저장: {simulation_file}")
        
        # 4. 분석
        print("\n3. 분석 실행 중...")
        balance_analysis = analyze_regional_balance_indicators(bds_data)
        priority_analysis = analyze_investment_priority(bds_data, target_regions)
        
        # 5. 검증 (선택적)
        validation_results = None
        investment_results = None
        if not args.skip_validation:
            print("\n4. 검증 실행 중...")
            validation_results = validate_improvement_potential(bds_data)
            investment_results = calculate_investment_impact(priority_analysis, args.investment_budget)
        
        # 6. 시각화 (선택적)
        if not args.skip_visualization:
            print("\n5. 시각화 생성 중...")
            
            # 기본 시뮬레이션 시각화
            sim_fig = visualize_simulation(simulation_data, target_regions)
            sim_fig.write_html(output_dir / "simulation.html")
            
            # 대시보드
            fig1, fig2, fig3, fig4 = create_simulation_dashboard(simulation_data, target_regions, balance_analysis)
            fig1.write_html(output_dir / "trend.html")
            fig2.write_html(output_dir / "final_comparison.html")
            fig3.write_html(output_dir / "improvement.html")
            fig4.write_html(output_dir / "balance_metrics.html")
            
            # 투자 우선순위
            if priority_analysis:
                priority_fig1, priority_fig2 = create_investment_priority_visualization(priority_analysis)
                priority_fig1.write_html(output_dir / "investment_priority.html")
                priority_fig2.write_html(output_dir / "improvement_potential.html")
            
            # 검증 결과
            if validation_results and investment_results:
                val_fig1, val_fig2 = create_validation_visualization(validation_results['validation_results'], investment_results)
                val_fig1.write_html(output_dir / "prediction_validation.html")
                val_fig2.write_html(output_dir / "investment_impact.html")
            
            # 요약
            sum_fig1, sum_fig2 = create_summary_visualization(bds_data, target_regions)
            sum_fig1.write_html(output_dir / "current_situation.html")
            sum_fig2.write_html(output_dir / "regional_indicators.html")
            
            # NAVIS 비교
            if navis_data:
                comparison_results = compare_bds_with_navis(bds_data, navis_data, target_year=2019)
                if comparison_results:
                    comp_fig = visualize_comparison(comparison_results, target_year=2019)
                    if comp_fig:
                        comp_fig.write_html(output_dir / "navis_comparison.html")
        
        # 7. 보고서 생성
        print("\n6. 보고서 생성 중...")
        comparison_results = compare_bds_with_navis(bds_data, navis_data, target_year=2019) if navis_data else {}
        report = generate_analysis_report(bds_data, navis_data, comparison_results, balance_analysis)
        
        # 시뮬레이션 파라미터 정보 추가
        report += f"\n## 시뮬레이션 파라미터\n\n"
        report += f"- 기본 개선률: {args.improvement_rate:.3f}\n"
        report += f"- 개선 잠재력 배수: {args.potential_multiplier:.3f}\n"
        report += f"- 체감 효과 계수: {args.diminishing_factor:.3f}\n"
        report += f"- 투자 예산: {args.investment_budget:,.0f}억원\n"
        report += f"- 시뮬레이션 기간: {args.simulation_years}년\n"
        report += f"- 설명: {args.description}\n\n"
        
        # 투자 우선순위 정보 추가
        if priority_analysis:
            report += "\n## 투자 우선순위 분석\n\n"
            for i, item in enumerate(priority_analysis, 1):
                report += f"### {i}순위: {item['region']}\n"
                report += f"- BDS 점수: {item['bds_score']:.3f}\n"
                report += f"- 투자 우선순위 점수: {item['overall_priority']:.3f}\n"
                report += f"- 개선 잠재력: {item['improvement_potential']:.3f}\n\n"
        
        # 검증 결과 추가
        if validation_results:
            report += "\n## 검증 결과\n\n"
            report += f"- 개선 잠재력 지표 유효성: {validation_results['avg_correlation']:.4f}\n"
            report += f"- 예측 정확도: {validation_results['validation_results']['correlation']:.4f}\n\n"
        
        # 투자 효과 추가
        if investment_results:
            report += "\n## 투자 효과 분석\n\n"
            total_roi = sum([r['roi'] for r in investment_results]) / len(investment_results)
            total_economic_effect = sum([r['total_economic_effect'] for r in investment_results])
            report += f"- 평균 ROI: {total_roi:.1f}%\n"
            report += f"- 총 경제 효과: {total_economic_effect/100000000:,.0f}억원\n\n"
        
        # 보고서 저장
        report_file = output_dir / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 8. 결과 요약
        print("\n=== 시뮬레이션 완료 ===")
        print(f"출력 디렉토리: {output_dir}")
        print(f"대상 지역: {', '.join(target_regions)}")
        print(f"파라미터: 개선률={args.improvement_rate:.3f}, 잠재력배수={args.potential_multiplier:.3f}")
        
        if validation_results:
            print(f"예측 정확도: {validation_results['validation_results']['correlation']:.4f}")
        
        if investment_results:
            total_roi = sum([r['roi'] for r in investment_results]) / len(investment_results)
            print(f"평균 ROI: {total_roi:.1f}%")
        
        # 생성된 파일 목록
        files = list(output_dir.glob("*.html")) + list(output_dir.glob("*.csv")) + list(output_dir.glob("*.json")) + list(output_dir.glob("*.md"))
        print(f"\n생성된 파일 수: {len(files)}")
        
        return True
        
    except Exception as e:
        print(f"시뮬레이션 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
    args = parse_arguments()
    success = run_simulation(args)
    
    if success:
        print("\n✅ 시뮬레이션 성공적으로 완료")
        sys.exit(0)
    else:
        print("\n❌ 시뮬레이션 실행 실패")
        sys.exit(1)

if __name__ == "__main__":
    main() 