#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
재정자립도 정책 시뮬레이터
재정자립도 낮은 지역(30% 미만)에 대한 정책 효과 시뮬레이션
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fiscal_policy_simulation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class FiscalPolicySimulator:
    def __init__(self):
        self.db_path = 'fiscal_autonomy_data.db'
        self.target_regions = ['전라북도', '경상북도', '전라남도', '강원도']  # 30% 미만 지역
        self.current_year = 2025
        self.simulation_years = 5  # 2025-2030년
        
        # 정책 효과 계수 (선행연구 기반, 현실적 조정)
        self.policy_effects = {
            'population_support': {
                'youth_settlement': 0.03,  # 청년 정착 지원 효과
                'elderly_welfare': 0.02,   # 고령자 복지 강화 효과
                'migration_incentive': 0.025 # 이주민 유치 효과
            },
            'industry_development': {
                'smart_farming': 0.04,     # 스마트팜 육성 효과
                'manufacturing': 0.05,     # 제조업 유치 효과
                'tourism': 0.035,         # 관광산업 육성 효과
                'digital_services': 0.045  # 디지털 서비스업 효과
            },
            'infrastructure': {
                'transport': 0.06,         # 교통 인프라 효과
                'digital': 0.05,          # 디지털 인프라 효과
                'logistics': 0.04         # 물류 인프라 효과
            },
            'institutional': {
                'fiscal_autonomy': 0.07,   # 재정 자율성 확대 효과
                'tax_diversification': 0.055, # 세원 다양화 효과
                'local_empowerment': 0.04   # 지방자치 강화 효과
            }
        }
        
    def load_current_data(self):
        """현재 재정자립도 데이터 로드"""
        try:
            df = pd.read_csv('kosis_fiscal_autonomy_data.csv')
            current_data = df[df['year'] == self.current_year].copy()
            
            # 대상 지역 데이터 추출
            target_data = current_data[current_data['region'].isin(self.target_regions)].copy()
            
            logging.info(f"현재 데이터 로드 완료: {len(target_data)}개 지역")
            return target_data
            
        except Exception as e:
            logging.error(f"데이터 로드 실패: {e}")
            return None
    
    def simulate_population_policy(self, region_data: pd.DataFrame, policy_intensity: float = 1.0):
        """인구 정책 시뮬레이션"""
        results = []
        
        for _, row in region_data.iterrows():
            region = row['region']
            current_ratio = row['fiscal_autonomy_ratio']
            
            # 인구 정책 효과 계산
            youth_effect = self.policy_effects['population_support']['youth_settlement'] * policy_intensity
            elderly_effect = self.policy_effects['population_support']['elderly_welfare'] * policy_intensity
            migration_effect = self.policy_effects['population_support']['migration_incentive'] * policy_intensity
            
            total_population_effect = (youth_effect + elderly_effect + migration_effect) * 0.02
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                # 누적 효과 적용
                cumulative_effect = total_population_effect * (year - self.current_year)
                new_ratio = min(0.95, current_ratio + cumulative_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'policy_type': 'population_support',
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'policy_intensity': policy_intensity
                })
        
        return pd.DataFrame(results)
    
    def simulate_industry_policy(self, region_data: pd.DataFrame, policy_intensity: float = 1.0):
        """산업 정책 시뮬레이션"""
        results = []
        
        for _, row in region_data.iterrows():
            region = row['region']
            current_ratio = row['fiscal_autonomy_ratio']
            
            # 산업별 특화 정책 효과
            if region in ['전라북도', '전라남도']:
                # 농업 특화 지역
                smart_farming_effect = self.policy_effects['industry_development']['smart_farming'] * policy_intensity
                tourism_effect = self.policy_effects['industry_development']['tourism'] * policy_intensity * 0.8
                total_industry_effect = (smart_farming_effect + tourism_effect) * 0.03
            else:
                # 제조업/서비스업 육성 지역
                manufacturing_effect = self.policy_effects['industry_development']['manufacturing'] * policy_intensity
                digital_effect = self.policy_effects['industry_development']['digital_services'] * policy_intensity
                total_industry_effect = (manufacturing_effect + digital_effect) * 0.03
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                cumulative_effect = total_industry_effect * (year - self.current_year)
                new_ratio = min(0.95, current_ratio + cumulative_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'policy_type': 'industry_development',
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'policy_intensity': policy_intensity
                })
        
        return pd.DataFrame(results)
    
    def simulate_infrastructure_policy(self, region_data: pd.DataFrame, policy_intensity: float = 1.0):
        """인프라 정책 시뮬레이션"""
        results = []
        
        for _, row in region_data.iterrows():
            region = row['region']
            current_ratio = row['fiscal_autonomy_ratio']
            
            # 인프라 정책 효과
            transport_effect = self.policy_effects['infrastructure']['transport'] * policy_intensity
            digital_effect = self.policy_effects['infrastructure']['digital'] * policy_intensity
            logistics_effect = self.policy_effects['infrastructure']['logistics'] * policy_intensity
            
            total_infrastructure_effect = (transport_effect + digital_effect + logistics_effect) * 0.025
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                cumulative_effect = total_infrastructure_effect * (year - self.current_year)
                new_ratio = min(0.95, current_ratio + cumulative_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'policy_type': 'infrastructure_investment',
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'policy_intensity': policy_intensity
                })
        
        return pd.DataFrame(results)
    
    def simulate_institutional_policy(self, region_data: pd.DataFrame, policy_intensity: float = 1.0):
        """제도적 정책 시뮬레이션"""
        results = []
        
        for _, row in region_data.iterrows():
            region = row['region']
            current_ratio = row['fiscal_autonomy_ratio']
            
            # 제도적 정책 효과
            fiscal_autonomy_effect = self.policy_effects['institutional']['fiscal_autonomy'] * policy_intensity
            tax_diversification_effect = self.policy_effects['institutional']['tax_diversification'] * policy_intensity
            local_empowerment_effect = self.policy_effects['institutional']['local_empowerment'] * policy_intensity
            
            total_institutional_effect = (fiscal_autonomy_effect + tax_diversification_effect + local_empowerment_effect) * 0.015
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                cumulative_effect = total_institutional_effect * (year - self.current_year)
                new_ratio = min(0.95, current_ratio + cumulative_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'policy_type': 'institutional_reform',
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'policy_intensity': policy_intensity
                })
        
        return pd.DataFrame(results)
    
    def simulate_comprehensive_policy(self, region_data: pd.DataFrame, policy_intensity: float = 1.0):
        """종합 정책 시뮬레이션 (모든 정책 동시 적용)"""
        results = []
        
        for _, row in region_data.iterrows():
            region = row['region']
            current_ratio = row['fiscal_autonomy_ratio']
            
            # 모든 정책 효과 종합
            population_effect = sum(self.policy_effects['population_support'].values()) * 0.02
            industry_effect = sum(self.policy_effects['industry_development'].values()) * 0.03
            infrastructure_effect = sum(self.policy_effects['infrastructure'].values()) * 0.025
            institutional_effect = sum(self.policy_effects['institutional'].values()) * 0.015
            
            # 시너지 효과 (정책 간 상호작용)
            synergy_effect = 0.005
            
            total_comprehensive_effect = (population_effect + industry_effect + infrastructure_effect + institutional_effect + synergy_effect) * policy_intensity
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                cumulative_effect = total_comprehensive_effect * (year - self.current_year)
                new_ratio = min(0.95, current_ratio + cumulative_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'policy_type': 'comprehensive_policy',
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'policy_intensity': policy_intensity
                })
        
        return pd.DataFrame(results)
    
    def run_simulation(self, policy_scenarios: Dict[str, float] = None):
        """정책 시뮬레이션 실행"""
        if policy_scenarios is None:
            policy_scenarios = {
                'population_support': 1.0,
                'industry_development': 1.0,
                'infrastructure_investment': 1.0,
                'institutional_reform': 1.0,
                'comprehensive_policy': 1.0
            }
        
        # 현재 데이터 로드
        current_data = self.load_current_data()
        if current_data is None:
            return None
        
        all_results = []
        
        # 각 정책 시나리오별 시뮬레이션
        for policy_type, intensity in policy_scenarios.items():
            if policy_type == 'population_support':
                results = self.simulate_population_policy(current_data, intensity)
            elif policy_type == 'industry_development':
                results = self.simulate_industry_policy(current_data, intensity)
            elif policy_type == 'infrastructure_investment':
                results = self.simulate_infrastructure_policy(current_data, intensity)
            elif policy_type == 'institutional_reform':
                results = self.simulate_institutional_policy(current_data, intensity)
            elif policy_type == 'comprehensive_policy':
                results = self.simulate_comprehensive_policy(current_data, intensity)
            else:
                continue
            
            all_results.append(results)
        
        # 결과 통합
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # 결과 저장
        combined_results.to_csv('fiscal_policy_simulation_results.csv', index=False, encoding='utf-8')
        
        logging.info(f"시뮬레이션 완료: {len(combined_results)}개 결과 생성")
        return combined_results
    
    def create_simulation_charts(self, results: pd.DataFrame):
        """시뮬레이션 결과 차트 생성"""
        
        # 1. 정책별 효과 비교 차트
        fig1 = go.Figure()
        
        for policy_type in results['policy_type'].unique():
            policy_data = results[results['policy_type'] == policy_type]
            avg_improvement = policy_data.groupby('year')['improvement'].mean()
            
            fig1.add_trace(go.Scatter(
                x=avg_improvement.index,
                y=avg_improvement.values * 100,
                mode='lines+markers',
                name=policy_type.replace('_', ' ').title(),
                line=dict(width=3)
            ))
        
        fig1.update_layout(
            title='정책별 재정자립도 개선 효과 (평균)',
            xaxis_title='연도',
            yaxis_title='재정자립도 개선폭 (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # 2. 지역별 종합 정책 효과 차트
        comprehensive_data = results[results['policy_type'] == 'comprehensive_policy']
        
        fig2 = go.Figure()
        
        for region in comprehensive_data['region'].unique():
            region_data = comprehensive_data[comprehensive_data['region'] == region]
            
            fig2.add_trace(go.Scatter(
                x=region_data['year'],
                y=region_data['new_ratio'] * 100,
                mode='lines+markers',
                name=region,
                line=dict(width=3)
            ))
        
        fig2.update_layout(
            title='지역별 종합 정책 효과 (재정자립도 변화)',
            xaxis_title='연도',
            yaxis_title='재정자립도 (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # 3. 정책 강도별 효과 차트
        fig3 = go.Figure()
        
        intensity_levels = [0.5, 1.0, 1.5, 2.0]
        colors = ['lightblue', 'blue', 'darkblue', 'navy']
        
        for i, intensity in enumerate(intensity_levels):
            # 종합 정책의 강도별 효과 시뮬레이션
            current_data = self.load_current_data()
            if current_data is not None:
                intensity_results = self.simulate_comprehensive_policy(current_data, intensity)
                avg_improvement = intensity_results.groupby('year')['improvement'].mean()
                
                fig3.add_trace(go.Scatter(
                    x=avg_improvement.index,
                    y=avg_improvement.values * 100,
                    mode='lines+markers',
                    name=f'정책 강도 {intensity}x',
                    line=dict(color=colors[i], width=3)
                ))
        
        fig3.update_layout(
            title='정책 강도별 효과 비교',
            xaxis_title='연도',
            yaxis_title='재정자립도 개선폭 (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # HTML 파일로 저장
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>재정자립도 정책 시뮬레이션 결과</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ margin: 30px 0; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>재정자립도 정책 시뮬레이션 결과</h1>
            <div class="summary">
                <h2>시뮬레이션 개요</h2>
                <p><strong>분석 대상:</strong> 재정자립도 30% 미만 지역 (전라북도, 경상북도, 전라남도, 강원도)</p>
                <p><strong>시뮬레이션 기간:</strong> {self.current_year}년 ~ {self.current_year + self.simulation_years}년</p>
                <p><strong>정책 유형:</strong> 인구 정책, 산업 정책, 인프라 정책, 제도적 정책, 종합 정책</p>
            </div>
            
            <div class="chart-container">
                <div id="chart1"></div>
            </div>
            
            <div class="chart-container">
                <div id="chart2"></div>
            </div>
            
            <div class="chart-container">
                <div id="chart3"></div>
            </div>
            
            <script>
                {fig1.to_json()}
                {fig2.to_json()}
                {fig3.to_json()}
                
                Plotly.newPlot('chart1', fig1.data, fig1.layout);
                Plotly.newPlot('chart2', fig2.data, fig2.layout);
                Plotly.newPlot('chart3', fig3.data, fig3.layout);
            </script>
        </body>
        </html>
        """
        
        with open('fiscal_policy_simulation_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info("시뮬레이션 차트 생성 완료")
        return fig1, fig2, fig3

def main():
    """메인 실행 함수"""
    simulator = FiscalPolicySimulator()
    
    # 기본 시뮬레이션 실행
    results = simulator.run_simulation()
    
    if results is not None:
        # 차트 생성
        simulator.create_simulation_charts(results)
        
        # 결과 요약 출력
        print("\n=== 재정자립도 정책 시뮬레이션 결과 요약 ===")
        
        # 2030년 종합 정책 효과
        final_results = results[results['year'] == 2030]
        comprehensive_final = final_results[final_results['policy_type'] == 'comprehensive_policy']
        
        print(f"\n2030년 종합 정책 효과:")
        for _, row in comprehensive_final.iterrows():
            improvement_pct = row['improvement'] * 100
            new_ratio_pct = row['new_ratio'] * 100
            print(f"  {row['region']}: {new_ratio_pct:.1f}% (개선폭: +{improvement_pct:.1f}%)")
        
        # 정책별 평균 효과
        print(f"\n정책별 평균 개선 효과 (2030년):")
        for policy_type in results['policy_type'].unique():
            policy_final = final_results[final_results['policy_type'] == policy_type]
            avg_improvement = policy_final['improvement'].mean() * 100
            print(f"  {policy_type.replace('_', ' ').title()}: +{avg_improvement:.1f}%")
        
        print(f"\n시뮬레이션 결과가 'fiscal_policy_simulation_results.csv'에 저장되었습니다.")
        print(f"대시보드는 'fiscal_policy_simulation_dashboard.html'에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
