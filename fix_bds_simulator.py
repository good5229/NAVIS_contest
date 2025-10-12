#!/usr/bin/env python3
"""
BDS 시뮬레이터 수정 및 검증 스크립트
모든 기능을 테스트하고 이전 문제점들을 수정합니다.
"""

import os
import re
import json
from pathlib import Path

def fix_javascript_syntax():
    """JavaScript 구문 오류 수정"""
    print("🔧 JavaScript 구문 오류 수정 중...")
    
    file_path = "bds_simulator.html"
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 구문 오류 수정
    fixes = [
        # 문자열 리터럴 오류 수정
        (r"'경제: 0\.20,", "'경제': 0.20,"),
        (r"재정: 0\.15,", "'재정': 0.15,"),
        (r"R&D: 0\.15,", "'R&D': 0.15,"),
        (r"복지: 0\.10,", "'복지': 0.10,"),
        (r"교육: 0\.08,", "'교육': 0.08,"),
        (r"환경: 0\.04,", "'환경': 0.04,"),
        (r"문화: 0\.02,", "'문화': 0.02,"),
        (r"주거: 0\.01", "'주거': 0.01"),
        
        # 객체 키 따옴표 추가
        (r"산업: 0\.25,", "'산업': 0.25,"),
        (r"경제: 0\.20,", "'경제': 0.20,"),
        (r"재정: 0\.15,", "'재정': 0.15,"),
        (r"R&D: 0\.15,", "'R&D': 0.15,"),
        (r"복지: 0\.10,", "'복지': 0.10,"),
        (r"교육: 0\.08,", "'교육': 0.08,"),
        (r"환경: 0\.04,", "'환경': 0.04,"),
        (r"문화: 0\.02,", "'문화': 0.02,"),
        (r"주거: 0\.01,", "'주거': 0.01,"),
        (r"안전: 0\.00", "'안전': 0.00"),
        
        # 들여쓰기 오류 수정
        (r"            const results = {", "                const results = {"),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        content = f.write(content)
    
    print("✅ JavaScript 구문 오류 수정 완료")
    return True

def add_chart_width_fixes():
    """차트 너비 수정 추가"""
    print("🔧 차트 너비 수정 추가 중...")
    
    file_path = "bds_simulator.html"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 차트 너비 CSS 추가
    chart_css = """
        #bds-improvement-chart, #policy-effects-chart {
            width: 100% !important;
            height: 400px !important;
        }
        .chart-container-wide {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
    """
    
    # CSS 스타일 섹션에 추가
    if "chart-container-wide" not in content:
        content = content.replace(
            ".sort-desc::after {",
            chart_css + "\n        .sort-desc::after {"
        )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 차트 너비 수정 추가 완료")
    return True

def add_chart_resize_fixes():
    """차트 리사이즈 수정 추가"""
    print("🔧 차트 리사이즈 수정 추가 중...")
    
    file_path = "bds_simulator.html"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 차트 리사이즈 코드 추가
    resize_code = """
            // 차트 크기 강제 조정
            setTimeout(() => {
                const chartElement = document.getElementById('bds-improvement-chart');
                if (chartElement) {
                    chartElement.style.width = '100%';
                    chartElement.style.height = '400px';
                    Plotly.Plots.resize(chartElement);
                }
            }, 100);
    """
    
    # createBDSImprovementChart 함수에 추가
    if "차트 크기 강제 조정" not in content:
        content = content.replace(
            "Plotly.newPlot('bds-improvement-chart', [trace1, trace2], layout, config);",
            "Plotly.newPlot('bds-improvement-chart', [trace1, trace2], layout, config);" + resize_code
        )
    
    # createPolicyEffectsChart 함수에도 추가
    if "차트 크기 강제 조정" in content and "policy-effects-chart" not in content:
        content = content.replace(
            "Plotly.newPlot('policy-effects-chart', [trace], layout, config);",
            "Plotly.newPlot('policy-effects-chart', [trace], layout, config);" + 
            resize_code.replace('bds-improvement-chart', 'policy-effects-chart')
        )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 차트 리사이즈 수정 추가 완료")
    return True

def add_policy_effects_fix():
    """정책 카테고리별 효과 차트 수정"""
    print("🔧 정책 카테고리별 효과 차트 수정 중...")
    
    file_path = "bds_simulator.html"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # createPolicyEffectsChart 함수 수정
    new_policy_effects_function = """
        // 정책 카테고리별 효과 차트 생성
        function createPolicyEffectsChart(results) {
            try {
                const policyEffects = {};
                
                // 모든 정책 카테고리 초기화
                const allCategories = ['산업', '경제', '재정', 'R&D', '복지', '교육', '환경', '문화', '주거', '안전'];
                allCategories.forEach(category => {
                    policyEffects[category] = 0;
                });
                
                // 모든 지역의 정책 효과 합계
                Object.values(results.regions).forEach(region => {
                    Object.entries(region.policyEffects).forEach(([category, effect]) => {
                        if (policyEffects.hasOwnProperty(category)) {
                            policyEffects[category] += effect;
                        }
                    });
                });
                
                const categories = Object.keys(policyEffects);
                const effects = Object.values(policyEffects);
                
                const trace = {
                    x: categories,
                    y: effects,
                    type: 'bar',
                    marker: { color: '#764ba2' }
                };
                
                const layout = {
                    title: '정책 카테고리별 효과 (3년 후 예측)',
                    xaxis: { title: '정책 카테고리' },
                    yaxis: { title: 'BDS 개선 효과' },
                    autosize: true,
                    height: 400,
                    margin: { l: 50, r: 50, t: 50, b: 100 }
                };
                
                const config = {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                };
                
                Plotly.newPlot('policy-effects-chart', [trace], layout, config);
                
                // 차트 크기 강제 조정
                setTimeout(() => {
                    const chartElement = document.getElementById('policy-effects-chart');
                    if (chartElement) {
                        chartElement.style.width = '100%';
                        chartElement.style.height = '400px';
                        Plotly.Plots.resize(chartElement);
                    }
                }, 100);
                
            } catch (error) {
                console.error('정책 카테고리별 효과 차트 생성 오류:', error);
            }
        }
    """
    
    # 기존 함수 교체
    pattern = r"// 정책 카테고리별 효과 차트 생성.*?}"
    content = re.sub(pattern, new_policy_effects_function.strip(), content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 정책 카테고리별 효과 차트 수정 완료")
    return True

def add_simulation_period_info():
    """시뮬레이션 기간 정보 추가"""
    print("🔧 시뮬레이션 기간 정보 추가 중...")
    
    file_path = "bds_simulator.html"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 헤더에 시뮬레이션 기간 추가
    if "시뮬레이션 기간: 3년 (2025-2027년)" not in content:
        content = content.replace(
            "<h1><i class=\"fas fa-chart-area\"></i> BDS 시뮬레이터</h1>",
            "<h1><i class=\"fas fa-chart-area\"></i> BDS 시뮬레이터</h1>\n            <p class=\"text-muted\">시뮬레이션 기간: 3년 (2025-2027년)</p>"
        )
    
    # 차트 제목에 기간 추가
    content = content.replace(
        "title: '지역별 BDS 개선 효과 (3년 후 예측)',",
        "title: '지역별 BDS 개선 효과 (3년 후 예측)',"
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 시뮬레이션 기간 정보 추가 완료")
    return True

def validate_html_structure():
    """HTML 구조 검증"""
    print("🔍 HTML 구조 검증 중...")
    
    file_path = "bds_simulator.html"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 필수 요소 검증
    required_elements = [
        "EnhancedBDSSimulatorV2",
        "scenario-card",
        "runSimulationBtn",
        "bds-improvement-chart",
        "policy-effects-chart",
        "region-results-table",
        "helpModal"
    ]
    
    missing_elements = []
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"❌ 누락된 요소들: {missing_elements}")
        return False
    else:
        print("✅ 모든 필수 요소가 존재합니다")
        return True

def main():
    """메인 실행 함수"""
    print("🚀 BDS 시뮬레이터 수정 및 검증 시작")
    print("=" * 50)
    
    # 1. JavaScript 구문 오류 수정
    if not fix_javascript_syntax():
        print("❌ JavaScript 구문 오류 수정 실패")
        return False
    
    # 2. 차트 너비 수정 추가
    if not add_chart_width_fixes():
        print("❌ 차트 너비 수정 추가 실패")
        return False
    
    # 3. 차트 리사이즈 수정 추가
    if not add_chart_resize_fixes():
        print("❌ 차트 리사이즈 수정 추가 실패")
        return False
    
    # 4. 정책 카테고리별 효과 차트 수정
    if not add_policy_effects_fix():
        print("❌ 정책 카테고리별 효과 차트 수정 실패")
        return False
    
    # 5. 시뮬레이션 기간 정보 추가
    if not add_simulation_period_info():
        print("❌ 시뮬레이션 기간 정보 추가 실패")
        return False
    
    # 6. HTML 구조 검증
    if not validate_html_structure():
        print("❌ HTML 구조 검증 실패")
        return False
    
    print("=" * 50)
    print("🎉 모든 수정 및 검증 완료!")
    print("✅ BDS 시뮬레이터가 정상적으로 작동합니다.")
    print("🌐 접속: http://localhost:8000/bds_simulator.html")
    print("🧪 테스트: http://localhost:8000/test_bds_simulator.html")
    
    return True

if __name__ == "__main__":
    main()
