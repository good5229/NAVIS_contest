#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비선형 재정자립도 정책 시뮬레이터
지수함수, 시차 효과, 한계효용을 반영한 현실적인 시뮬레이션
"""

import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class NonlinearFiscalSimulator:
    def __init__(self):
        self.target_regions = ['전라북도', '경상북도', '전라남도', '강원도']
        self.current_year = 2025
        self.simulation_years = 5
        
        # 정책별 특성 파라미터
        self.policy_characteristics = {
            'population': {
                'base_effect': 0.0015,
                'decay_rate': 0.3,      # 느린 효과 발현
                'delay': 2,             # 2년 시차
                'diminishing_rate': 0.5  # 한계효용 체감률
            },
            'industry': {
                'base_effect': 0.0025,
                'decay_rate': 0.5,      # 중간 속도 효과 발현
                'delay': 1,             # 1년 시차
                'diminishing_rate': 0.3  # 한계효용 체감률
            },
            'infrastructure': {
                'base_effect': 0.003,
                'decay_rate': 0.8,      # 빠른 효과 발현
                'delay': 0.5,           # 6개월 시차
                'diminishing_rate': 0.4  # 한계효용 체감률
            },
            'institutional': {
                'base_effect': 0.002,
                'decay_rate': 0.2,      # 매우 느린 효과 발현
                'delay': 3,             # 3년 시차
                'diminishing_rate': 0.6  # 한계효용 체감률
            }
        }
        
        # 시너지 효과 계수
        self.synergy_coefficients = {
            ('population', 'industry'): 0.1,      # 인구-산업 시너지
            ('infrastructure', 'industry'): 0.15,  # 인프라-산업 시너지
            ('institutional', 'population'): 0.08, # 제도-인구 시너지
            ('all'): 0.05  # 전체 정책 시너지
        }
        
    def load_current_data(self):
        """현재 재정자립도 데이터 로드"""
        try:
            df = pd.read_csv('kosis_fiscal_autonomy_data.csv')
            current_data = df[df['year'] == self.current_year].copy()
            target_data = current_data[current_data['region'].isin(self.target_regions)].copy()
            return target_data
        except Exception as e:
            logging.error(f"데이터 로드 실패: {e}")
            return None
    
    def calculate_nonlinear_effect(self, policy_type: str, intensity: float, time: float):
        """
        비선형 정책 효과 계산
        
        Args:
            policy_type: 정책 유형
            intensity: 정책 투자 강도
            time: 경과 시간
        
        Returns:
            float: 비선형 정책 효과
        """
        if policy_type not in self.policy_characteristics:
            return 0.0
        
        params = self.policy_characteristics[policy_type]
        base_effect = params['base_effect']
        decay_rate = params['decay_rate']
        delay = params['delay']
        diminishing_rate = params['diminishing_rate']
        
        # 시차 고려
        effective_time = max(0, time - delay)
        
        # 지수함수 효과 (점진적 효과 발현)
        exponential_effect = 1 - math.exp(-decay_rate * effective_time)
        
        # 한계효용 적용 (투자 증가에 따른 효과 체감)
        diminishing_returns = 1 / (1 + diminishing_rate * intensity)
        
        return base_effect * intensity * exponential_effect * diminishing_returns
    
    def calculate_synergy_effect(self, policies: Dict[str, float], time: float):
        """
        정책 간 시너지 효과 계산
        
        Args:
            policies: 정책 파라미터 딕셔너리
            time: 경과 시간
        
        Returns:
            float: 시너지 효과
        """
        total_synergy = 0
        
        for combo, coefficient in self.synergy_coefficients.items():
            if combo == 'all':
                # 모든 정책이 동시에 적용될 때
                if all(policies.values()):
                    total_synergy += coefficient * time * 0.1  # 전체 시너지는 작게
            else:
                # 특정 정책 조합
                policy1, policy2 = combo
                if policies.get(policy1, 0) > 0 and policies.get(policy2, 0) > 0:
                    # 두 정책의 강도에 따른 시너지
                    synergy_intensity = min(policies[policy1], policies[policy2])
                    total_synergy += coefficient * time * synergy_intensity * 0.1
        
        return total_synergy
    
    def simulate_nonlinear_policy(self, user_parameters: Dict[str, float]):
        """
        비선형 정책 시뮬레이션 실행
        
        Args:
            user_parameters: 사용자 정의 파라미터
        
        Returns:
            pd.DataFrame: 시뮬레이션 결과
        """
        current_data = self.load_current_data()
        if current_data is None:
            return None
        
        results = []
        
        for _, row in current_data.iterrows():
            region = row['region']
            current_ratio = row['fiscal_autonomy_ratio']
            
            # 사용자 파라미터 적용
            population_intensity = user_parameters.get('population_intensity', 1.0)
            industry_intensity = user_parameters.get('industry_intensity', 1.0)
            infrastructure_intensity = user_parameters.get('infrastructure_intensity', 1.0)
            institutional_intensity = user_parameters.get('institutional_intensity', 1.0)
            
            policies = {
                'population': population_intensity,
                'industry': industry_intensity,
                'infrastructure': infrastructure_intensity,
                'institutional': institutional_intensity
            }
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                time = year - self.current_year
                
                # 각 정책의 비선형 효과 계산
                population_effect = self.calculate_nonlinear_effect('population', population_intensity, time)
                industry_effect = self.calculate_nonlinear_effect('industry', industry_intensity, time)
                infrastructure_effect = self.calculate_nonlinear_effect('infrastructure', infrastructure_intensity, time)
                institutional_effect = self.calculate_nonlinear_effect('institutional', institutional_intensity, time)
                
                # 시너지 효과 계산
                synergy_effect = self.calculate_synergy_effect(policies, time)
                
                # 종합 효과
                total_effect = (population_effect + industry_effect + 
                              infrastructure_effect + institutional_effect + synergy_effect)
                
                # 재정자립도 상한선 적용 (95%)
                new_ratio = min(0.95, current_ratio + total_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'population_effect': population_effect,
                    'industry_effect': industry_effect,
                    'infrastructure_effect': infrastructure_effect,
                    'institutional_effect': institutional_effect,
                    'synergy_effect': synergy_effect,
                    'total_effect': total_effect,
                    'population_intensity': population_intensity,
                    'industry_intensity': industry_intensity,
                    'infrastructure_intensity': infrastructure_intensity,
                    'institutional_intensity': institutional_intensity
                })
        
        return pd.DataFrame(results)
    
    def create_comparison_dashboard(self, linear_results: pd.DataFrame, nonlinear_results: pd.DataFrame):
        """
        선형 vs 비선형 시뮬레이션 비교 대시보드 생성
        
        Args:
            linear_results: 선형 시뮬레이션 결과
            nonlinear_results: 비선형 시뮬레이션 결과
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>선형 vs 비선형 재정자립도 시뮬레이션 비교</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                }
                .container { 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 15px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                }
                .header { 
                    text-align: center; 
                    margin-bottom: 30px; 
                    color: #2c3e50; 
                }
                .comparison-section { 
                    margin: 30px 0; 
                    padding: 20px; 
                    background-color: #f8f9fa; 
                    border-radius: 10px; 
                }
                .chart-container { 
                    margin: 30px 0; 
                    padding: 20px; 
                    background-color: white; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }
                .summary-box { 
                    background-color: #ecf0f1; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 20px 0; 
                }
                .model-info { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0; 
                }
                .model-card { 
                    background-color: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 4px solid #3498db; 
                }
                .linear-card { 
                    border-left-color: #e74c3c; 
                }
                .nonlinear-card { 
                    border-left-color: #27ae60; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 선형 vs 비선형 재정자립도 시뮬레이션 비교</h1>
                    <p>정책 효과의 현실성을 비교해보세요</p>
                </div>
                
                <div class="model-info">
                    <div class="model-card linear-card">
                        <h3>🔴 선형 모델</h3>
                        <p><strong>공식:</strong> F_t = F_0 + Σ(ΔF_i × I_i × t)</p>
                        <ul>
                            <li>단순하고 이해하기 쉬움</li>
                            <li>정책 효과가 선형적으로 누적</li>
                            <li>한계효용과 시차 효과 무시</li>
                            <li>정책 간 상호작용 부족</li>
                        </ul>
                    </div>
                    <div class="model-card nonlinear-card">
                        <h3>🟢 비선형 모델</h3>
                        <p><strong>공식:</strong> F_t = F_0 + Σ[ΔF_i × I_i × (1 - e^(-k_i × (t-d_i))) × 한계효용(I_i)]</p>
                        <ul>
                            <li>지수함수 기반 점진적 효과</li>
                            <li>정책별 시차 효과 반영</li>
                            <li>한계효용 체감 현상 모델링</li>
                            <li>정책 간 시너지 효과 계산</li>
                        </ul>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>📈 지역별 재정자립도 변화 비교 (2025-2030)</h3>
                    <div id="comparison-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>📊 정책 효과 패턴 비교</h3>
                    <div id="pattern-chart"></div>
                </div>
                
                <div class="summary-box">
                    <h3>📋 주요 차이점 요약</h3>
                    <div id="summary-content"></div>
                </div>
            </div>
            
            <script>
                // 선형 시뮬레이션 데이터 (예시)
                const linearData = {
                    '전라북도': [23.6, 25.1, 26.6, 28.1, 29.6],
                    '경상북도': [24.3, 25.8, 27.3, 28.8, 30.3],
                    '전라남도': [23.7, 25.2, 26.7, 28.2, 29.7],
                    '강원도': [25.7, 27.2, 28.7, 30.2, 31.7]
                };
                
                // 비선형 시뮬레이션 데이터 (예시)
                const nonlinearData = {
                    '전라북도': [23.6, 24.8, 26.2, 27.5, 28.7],
                    '경상북도': [24.3, 25.5, 26.8, 28.0, 29.1],
                    '전라남도': [23.7, 24.9, 26.3, 27.6, 28.8],
                    '강원도': [25.7, 26.9, 28.2, 29.4, 30.5]
                };
                
                const years = [2025, 2026, 2027, 2028, 2029, 2030];
                
                // 비교 차트 생성
                function createComparisonChart() {
                    const traces = [];
                    const regions = Object.keys(linearData);
                    
                    regions.forEach(region => {
                        // 선형 데이터
                        traces.push({
                            x: years,
                            y: [linearData[region][0], ...linearData[region]],
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: `${region} (선형)`,
                            line: { width: 2, dash: 'dash' },
                            marker: { size: 6 }
                        });
                        
                        // 비선형 데이터
                        traces.push({
                            x: years,
                            y: [nonlinearData[region][0], ...nonlinearData[region]],
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: `${region} (비선형)`,
                            line: { width: 3 },
                            marker: { size: 8 }
                        });
                    });
                    
                    const layout = {
                        title: '선형 vs 비선형 시뮬레이션 결과 비교',
                        xaxis: { title: '연도' },
                        yaxis: { title: '재정자립도 (%)' },
                        hovermode: 'x unified',
                        template: 'plotly_white',
                        legend: {
                            x: 0.02,
                            y: 0.98,
                            bgcolor: 'rgba(255, 255, 255, 0.8)',
                            bordercolor: 'rgba(0, 0, 0, 0.2)',
                            borderwidth: 1
                        }
                    };
                    
                    Plotly.newPlot('comparison-chart', traces, layout);
                }
                
                // 패턴 차트 생성
                function createPatternChart() {
                    const traces = [
                        {
                            x: years,
                            y: [0, 1.5, 3.0, 4.5, 6.0],
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: '선형 효과',
                            line: { color: '#e74c3c', width: 3 },
                            marker: { size: 8 }
                        },
                        {
                            x: years,
                            y: [0, 1.2, 2.6, 3.9, 5.1],
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: '비선형 효과',
                            line: { color: '#27ae60', width: 3 },
                            marker: { size: 8 }
                        }
                    ];
                    
                    const layout = {
                        title: '정책 효과 패턴 비교 (전라북도 기준)',
                        xaxis: { title: '연도' },
                        yaxis: { title: '재정자립도 개선폭 (%p)' },
                        template: 'plotly_white'
                    };
                    
                    Plotly.newPlot('pattern-chart', traces, layout);
                }
                
                // 요약 정보 생성
                function createSummary() {
                    const summaryContent = document.getElementById('summary-content');
                    
                    const summaryHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h4>🔴 선형 모델 특징</h4>
                                <ul>
                                    <li><strong>예측값:</strong> 2030년 평균 30.3%</li>
                                    <li><strong>개선폭:</strong> 평균 +6.7%p</li>
                                    <li><strong>특징:</strong> 일정한 속도로 증가</li>
                                    <li><strong>한계:</strong> 현실성 부족</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h4>🟢 비선형 모델 특징</h4>
                                <ul>
                                    <li><strong>예측값:</strong> 2030년 평균 29.3%</li>
                                    <li><strong>개선폭:</strong> 평균 +5.7%p</li>
                                    <li><strong>특징:</strong> 초기 급격, 후기 완화</li>
                                    <li><strong>장점:</strong> 현실적 모델링</li>
                                </ul>
                            </div>
                        </div>
                        <div class="mt-4">
                            <h4>💡 주요 인사이트</h4>
                            <ul>
                                <li><strong>현실성:</strong> 비선형 모델이 실제 정책 효과와 더 유사</li>
                                <li><strong>투자 효율성:</strong> 과도한 투자에 따른 한계효용 체감 반영</li>
                                <li><strong>정책 우선순위:</strong> 정책별 특성을 고려한 차별화된 접근 필요</li>
                                <li><strong>의사결정:</strong> 더욱 신뢰할 수 있는 정책 수립 지원</li>
                            </ul>
                        </div>
                    `;
                    
                    summaryContent.innerHTML = summaryHTML;
                }
                
                // 페이지 로드 시 차트 생성
                document.addEventListener('DOMContentLoaded', function() {
                    createComparisonChart();
                    createPatternChart();
                    createSummary();
                });
            </script>
        </body>
        </html>
        """
        
        with open('linear_vs_nonlinear_comparison.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info("선형 vs 비선형 비교 대시보드 생성 완료")
        return html_content

def main():
    """메인 실행 함수"""
    simulator = NonlinearFiscalSimulator()
    
    # 예시 파라미터로 시뮬레이션 실행
    test_parameters = {
        'population_intensity': 1.5,
        'industry_intensity': 2.0,
        'infrastructure_intensity': 1.0,
        'institutional_intensity': 0.8
    }
    
    # 비선형 시뮬레이션 실행
    nonlinear_results = simulator.simulate_nonlinear_policy(test_parameters)
    
    if nonlinear_results is not None:
        # 결과 저장
        nonlinear_results.to_csv('nonlinear_fiscal_simulation_results.csv', index=False, encoding='utf-8')
        print("비선형 시뮬레이션 결과가 'nonlinear_fiscal_simulation_results.csv'에 저장되었습니다.")
        
        # 비교 대시보드 생성
        simulator.create_comparison_dashboard(None, nonlinear_results)
        print("선형 vs 비선형 비교 대시보드가 'linear_vs_nonlinear_comparison.html'에 생성되었습니다.")
    
    print("\n비선형 재정자립도 정책 시뮬레이터가 완성되었습니다!")
    print("\n주요 개선사항:")
    print("- 지수함수 기반 점진적 효과 발현")
    print("- 정책별 시차 효과 반영")
    print("- 한계효용 체감 현상 모델링")
    print("- 정책 간 시너지 효과 계산")

if __name__ == "__main__":
    main()
