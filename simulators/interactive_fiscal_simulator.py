#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
인터랙티브 재정자립도 정책 시뮬레이터
사용자가 정책 파라미터를 직접 입력하여 시뮬레이션 실행
"""

import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class InteractiveFiscalSimulator:
    def __init__(self):
        self.target_regions = ['전라북도', '경상북도', '전라남도', '강원도']
        self.current_year = 2025
        self.simulation_years = 5
        
        # 기본 정책 효과 계수
        self.default_effects = {
            'population_support': {
                'youth_settlement': 0.03,
                'elderly_welfare': 0.02,
                'migration_incentive': 0.025
            },
            'industry_development': {
                'smart_farming': 0.04,
                'manufacturing': 0.05,
                'tourism': 0.035,
                'digital_services': 0.045
            },
            'infrastructure': {
                'transport': 0.06,
                'digital': 0.05,
                'logistics': 0.04
            },
            'institutional': {
                'fiscal_autonomy': 0.07,
                'tax_diversification': 0.055,
                'local_empowerment': 0.04
            }
        }
        
    def load_current_data(self):
        """현재 재정자립도 데이터 로드"""
        try:
            df = pd.read_csv('data/fiscal_autonomy/kosis_fiscal_autonomy_data.csv')
            current_data = df[df['year'] == self.current_year].copy()
            target_data = current_data[current_data['region'].isin(self.target_regions)].copy()
            return target_data
        except Exception as e:
            logging.error(f"데이터 로드 실패: {e}")
            return None
    
    def simulate_with_custom_parameters(self, user_parameters: Dict):
        """사용자 정의 파라미터로 시뮬레이션 실행"""
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
            synergy_multiplier = user_parameters.get('synergy_multiplier', 1.0)
            
            # 정책별 효과 계산
            population_effect = sum(self.default_effects['population_support'].values()) * 0.02 * population_intensity
            industry_effect = sum(self.default_effects['industry_development'].values()) * 0.03 * industry_intensity
            infrastructure_effect = sum(self.default_effects['infrastructure'].values()) * 0.025 * infrastructure_intensity
            institutional_effect = sum(self.default_effects['institutional'].values()) * 0.015 * institutional_intensity
            
            # 시너지 효과
            synergy_effect = 0.005 * synergy_multiplier
            
            # 종합 효과
            total_effect = (population_effect + industry_effect + infrastructure_effect + institutional_effect + synergy_effect)
            
            # 연도별 시뮬레이션
            for year in range(self.current_year + 1, self.current_year + self.simulation_years + 1):
                cumulative_effect = total_effect * (year - self.current_year)
                new_ratio = min(0.95, current_ratio + cumulative_effect)
                
                results.append({
                    'region': region,
                    'year': year,
                    'original_ratio': current_ratio,
                    'new_ratio': new_ratio,
                    'improvement': new_ratio - current_ratio,
                    'population_intensity': population_intensity,
                    'industry_intensity': industry_intensity,
                    'infrastructure_intensity': infrastructure_intensity,
                    'institutional_intensity': institutional_intensity,
                    'synergy_multiplier': synergy_multiplier
                })
        
        return pd.DataFrame(results)
    
    def create_interactive_dashboard(self):
        """인터랙티브 대시보드 생성"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>인터랙티브 재정자립도 정책 시뮬레이터</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                }
                .container { 
                    max-width: 1200px; 
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
                .parameter-section { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 30px; 
                    padding: 20px; 
                    background-color: #f8f9fa; 
                    border-radius: 10px; 
                }
                .parameter-group { 
                    background-color: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    border-left: 4px solid #3498db; 
                }
                .parameter-group h3 { 
                    margin-top: 0; 
                    color: #2c3e50; 
                    font-size: 18px; 
                }
                .slider-container { 
                    margin: 15px 0; 
                }
                .slider-container label { 
                    display: block; 
                    margin-bottom: 5px; 
                    font-weight: bold; 
                    color: #34495e; 
                }
                .slider { 
                    width: 100%; 
                    height: 6px; 
                    border-radius: 3px; 
                    background: #ddd; 
                    outline: none; 
                    -webkit-appearance: none; 
                }
                .slider::-webkit-slider-thumb { 
                    -webkit-appearance: none; 
                    appearance: none; 
                    width: 20px; 
                    height: 20px; 
                    border-radius: 50%; 
                    background: #3498db; 
                    cursor: pointer; 
                }
                .slider::-moz-range-thumb { 
                    width: 20px; 
                    height: 20px; 
                    border-radius: 50%; 
                    background: #3498db; 
                    cursor: pointer; 
                    border: none; 
                }
                .value-display { 
                    text-align: center; 
                    font-weight: bold; 
                    color: #e74c3c; 
                    font-size: 16px; 
                    margin-top: 5px; 
                }
                .button-section { 
                    text-align: center; 
                    margin: 30px 0; 
                }
                .simulate-btn { 
                    background-color: #27ae60; 
                    color: white; 
                    border: none; 
                    padding: 15px 30px; 
                    font-size: 18px; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    transition: background-color 0.3s; 
                }
                .simulate-btn:hover { 
                    background-color: #229954; 
                }
                .reset-btn { 
                    background-color: #e74c3c; 
                    color: white; 
                    border: none; 
                    padding: 15px 30px; 
                    font-size: 18px; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    margin-left: 15px; 
                    transition: background-color 0.3s; 
                }
                .reset-btn:hover { 
                    background-color: #c0392b; 
                }
                .results-section { 
                    margin-top: 30px; 
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
                .summary-box h3 { 
                    margin-top: 0; 
                    color: #2c3e50; 
                }
                .parameter-description { 
                    font-size: 14px; 
                    color: #7f8c8d; 
                    margin-top: 5px; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 인터랙티브 재정자립도 정책 시뮬레이터</h1>
                    <p>정책 파라미터를 조정하여 재정자립도 개선 효과를 시뮬레이션해보세요</p>
                </div>
                
                <div class="parameter-section">
                    <div class="parameter-group">
                        <h3>👥 인구 정책 강도</h3>
                        <div class="slider-container">
                            <label>청년 정착 지원</label>
                            <input type="range" id="population_intensity" class="slider" min="0" max="3" step="0.1" value="1.0">
                            <div class="value-display" id="population_intensity_value">1.0x</div>
                            <div class="parameter-description">청년 정착, 고령자 복지, 이주민 유치 정책 강도</div>
                        </div>
                    </div>
                    
                    <div class="parameter-group">
                        <h3>🏭 산업 정책 강도</h3>
                        <div class="slider-container">
                            <label>산업 육성 지원</label>
                            <input type="range" id="industry_intensity" class="slider" min="0" max="3" step="0.1" value="1.0">
                            <div class="value-display" id="industry_intensity_value">1.0x</div>
                            <div class="parameter-description">스마트팜, 제조업, 관광, 디지털 서비스업 육성</div>
                        </div>
                    </div>
                    
                    <div class="parameter-group">
                        <h3>🏗️ 인프라 투자 강도</h3>
                        <div class="slider-container">
                            <label>인프라 투자</label>
                            <input type="range" id="infrastructure_intensity" class="slider" min="0" max="3" step="0.1" value="1.0">
                            <div class="value-display" id="infrastructure_intensity_value">1.0x</div>
                            <div class="parameter-description">교통, 디지털, 물류 인프라 투자 강도</div>
                        </div>
                    </div>
                    
                    <div class="parameter-group">
                        <h3>⚖️ 제도적 개선 강도</h3>
                        <div class="slider-container">
                            <label>제도 개선</label>
                            <input type="range" id="institutional_intensity" class="slider" min="0" max="3" step="0.1" value="1.0">
                            <div class="value-display" id="institutional_intensity_value">1.0x</div>
                            <div class="parameter-description">재정 자율성, 세원 다양화, 지방자치 강화</div>
                        </div>
                    </div>
                    
                    <div class="parameter-group">
                        <h3>🔄 시너지 효과 배수</h3>
                        <div class="slider-container">
                            <label>정책 간 시너지</label>
                            <input type="range" id="synergy_multiplier" class="slider" min="0" max="3" step="0.1" value="1.0">
                            <div class="value-display" id="synergy_multiplier_value">1.0x</div>
                            <div class="parameter-description">여러 정책 동시 적용 시 상호작용 효과</div>
                        </div>
                    </div>
                </div>
                
                <div class="button-section">
                    <button class="simulate-btn" onclick="runSimulation()">🚀 시뮬레이션 실행</button>
                    <button class="reset-btn" onclick="resetParameters()">🔄 파라미터 초기화</button>
                </div>
                
                <div class="results-section" id="results-section" style="display: none;">
                    <div class="summary-box">
                        <h3>📊 시뮬레이션 결과 요약</h3>
                        <div id="summary-content"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>📈 지역별 재정자립도 변화 (2025-2030)</h3>
                        <div id="chart1"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>📊 정책 효과 분석</h3>
                        <div id="chart2"></div>
                    </div>
                </div>
            </div>
            
            <script>
                // 슬라이더 값 업데이트 함수
                function updateSliderValue(sliderId, valueId) {
                    const slider = document.getElementById(sliderId);
                    const valueDisplay = document.getElementById(valueId);
                    slider.oninput = function() {
                        valueDisplay.textContent = this.value + 'x';
                    };
                }
                
                // 모든 슬라이더 초기화
                updateSliderValue('population_intensity', 'population_intensity_value');
                updateSliderValue('industry_intensity', 'industry_intensity_value');
                updateSliderValue('infrastructure_intensity', 'infrastructure_intensity_value');
                updateSliderValue('institutional_intensity', 'institutional_intensity_value');
                updateSliderValue('synergy_multiplier', 'synergy_multiplier_value');
                
                // 파라미터 초기화
                function resetParameters() {
                    document.getElementById('population_intensity').value = 1.0;
                    document.getElementById('industry_intensity').value = 1.0;
                    document.getElementById('infrastructure_intensity').value = 1.0;
                    document.getElementById('institutional_intensity').value = 1.0;
                    document.getElementById('synergy_multiplier').value = 1.0;
                    
                    document.getElementById('population_intensity_value').textContent = '1.0x';
                    document.getElementById('industry_intensity_value').textContent = '1.0x';
                    document.getElementById('infrastructure_intensity_value').textContent = '1.0x';
                    document.getElementById('institutional_intensity_value').textContent = '1.0x';
                    document.getElementById('synergy_multiplier_value').textContent = '1.0x';
                }
                
                // 시뮬레이션 실행
                function runSimulation() {
                    const parameters = {
                        population_intensity: parseFloat(document.getElementById('population_intensity').value),
                        industry_intensity: parseFloat(document.getElementById('industry_intensity').value),
                        infrastructure_intensity: parseFloat(document.getElementById('infrastructure_intensity').value),
                        institutional_intensity: parseFloat(document.getElementById('institutional_intensity').value),
                        synergy_multiplier: parseFloat(document.getElementById('synergy_multiplier').value)
                    };
                    
                    // 시뮬레이션 실행 (실제로는 서버로 요청)
                    simulateWithParameters(parameters);
                }
                
                // 파라미터로 시뮬레이션 실행
                function simulateWithParameters(parameters) {
                    // 현재 데이터 (실제로는 서버에서 가져옴)
                    const currentData = {
                        '전라북도': 0.236,
                        '경상북도': 0.243,
                        '전라남도': 0.237,
                        '강원도': 0.257
                    };
                    
                    const results = [];
                    const regions = Object.keys(currentData);
                    
                    // 기본 효과 계수
                    const baseEffects = {
                        population: 0.0015,
                        industry: 0.0025,
                        infrastructure: 0.003,
                        institutional: 0.002,
                        synergy: 0.0005
                    };
                    
                    // 각 지역별 시뮬레이션
                    regions.forEach(region => {
                        const currentRatio = currentData[region];
                        
                        for (let year = 2026; year <= 2030; year++) {
                            const yearsDiff = year - 2025;
                            
                            // 정책별 효과 계산
                            const populationEffect = baseEffects.population * parameters.population_intensity * yearsDiff;
                            const industryEffect = baseEffects.industry * parameters.industry_intensity * yearsDiff;
                            const infrastructureEffect = baseEffects.infrastructure * parameters.infrastructure_intensity * yearsDiff;
                            const institutionalEffect = baseEffects.institutional * parameters.institutional_intensity * yearsDiff;
                            const synergyEffect = baseEffects.synergy * parameters.synergy_multiplier * yearsDiff;
                            
                            // 종합 효과
                            const totalEffect = populationEffect + industryEffect + infrastructureEffect + institutionalEffect + synergyEffect;
                            const newRatio = Math.min(0.95, currentRatio + totalEffect);
                            
                            results.push({
                                region: region,
                                year: year,
                                original_ratio: currentRatio,
                                new_ratio: newRatio,
                                improvement: newRatio - currentRatio,
                                parameters: parameters
                            });
                        }
                    });
                    
                    // 결과 표시
                    displayResults(results, parameters);
                }
                
                // 결과 표시
                function displayResults(results, parameters) {
                    const resultsSection = document.getElementById('results-section');
                    resultsSection.style.display = 'block';
                    
                    // 요약 정보 생성
                    const summaryContent = document.getElementById('summary-content');
                    const finalResults = results.filter(r => r.year === 2030);
                    
                    let summaryHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">';
                    
                    finalResults.forEach(result => {
                        const improvementPct = (result.improvement * 100).toFixed(1);
                        const newRatioPct = (result.new_ratio * 100).toFixed(1);
                        
                        summaryHTML += `
                            <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                                <h4 style="margin: 0 0 10px 0; color: #2c3e50;">${result.region}</h4>
                                <p style="margin: 5px 0; font-size: 14px;">현재: ${(result.original_ratio * 100).toFixed(1)}%</p>
                                <p style="margin: 5px 0; font-size: 14px;">2030년: <strong style="color: #e74c3c;">${newRatioPct}%</strong></p>
                                <p style="margin: 5px 0; font-size: 14px;">개선폭: <strong style="color: #27ae60;">+${improvementPct}%p</strong></p>
                            </div>
                        `;
                    });
                    
                    summaryHTML += '</div>';
                    
                    // 파라미터 정보 추가
                    summaryHTML += `
                        <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px;">
                            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📋 적용된 정책 파라미터</h4>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                                <div><strong>인구 정책:</strong> ${parameters.population_intensity}x</div>
                                <div><strong>산업 정책:</strong> ${parameters.industry_intensity}x</div>
                                <div><strong>인프라 정책:</strong> ${parameters.infrastructure_intensity}x</div>
                                <div><strong>제도적 정책:</strong> ${parameters.institutional_intensity}x</div>
                                <div><strong>시너지 효과:</strong> ${parameters.synergy_multiplier}x</div>
                            </div>
                        </div>
                    `;
                    
                    summaryContent.innerHTML = summaryHTML;
                    
                    // 차트 생성
                    createCharts(results);
                }
                
                // 차트 생성
                function createCharts(results) {
                    // 차트 1: 지역별 재정자립도 변화
                    const chart1Data = [];
                    const regions = [...new Set(results.map(r => r.region))];
                    
                    regions.forEach(region => {
                        const regionData = results.filter(r => r.region === region);
                        chart1Data.push({
                            x: regionData.map(r => r.year),
                            y: regionData.map(r => r.new_ratio * 100),
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: region,
                            line: { width: 3 }
                        });
                    });
                    
                    const chart1Layout = {
                        title: '지역별 재정자립도 변화 추이 (2025-2030)',
                        xaxis: { title: '연도' },
                        yaxis: { title: '재정자립도 (%)' },
                        hovermode: 'x unified',
                        template: 'plotly_white'
                    };
                    
                    Plotly.newPlot('chart1', chart1Data, chart1Layout);
                    
                    // 차트 2: 정책 효과 분석
                    const finalResults = results.filter(r => r.year === 2030);
                    const improvements = finalResults.map(r => r.improvement * 100);
                    const regionNames = finalResults.map(r => r.region);
                    
                    const chart2Data = [{
                        x: regionNames,
                        y: improvements,
                        type: 'bar',
                        marker: {
                            color: improvements.map(v => v > 10 ? '#27ae60' : v > 5 ? '#f39c12' : '#e74c3c')
                        }
                    }];
                    
                    const chart2Layout = {
                        title: '지역별 재정자립도 개선 효과 (2030년)',
                        xaxis: { title: '지역' },
                        yaxis: { title: '개선폭 (%p)' },
                        template: 'plotly_white'
                    };
                    
                    Plotly.newPlot('chart2', chart2Data, chart2Layout);
                }
            </script>
        </body>
        </html>
        """
        
        with open('interactive_fiscal_simulator.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info("인터랙티브 시뮬레이터 대시보드 생성 완료")
        return html_content

def main():
    """메인 실행 함수"""
    simulator = InteractiveFiscalSimulator()
    simulator.create_interactive_dashboard()
    
    print("인터랙티브 재정자립도 정책 시뮬레이터가 생성되었습니다!")
    print("'interactive_fiscal_simulator.html' 파일을 브라우저에서 열어 사용하세요.")
    print("\n주요 기능:")
    print("- 5가지 정책 파라미터 조정 (0x ~ 3x)")
    print("- 실시간 시뮬레이션 실행")
    print("- 지역별 결과 비교")
    print("- 정책 효과 시각화")

if __name__ == "__main__":
    main()
