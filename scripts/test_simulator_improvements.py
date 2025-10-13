#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시뮬레이터 시나리오별 개선 효과 테스트
- 각 시나리오의 개선 효과가 올바른지 검증
- 잘못된 값(70.38%)이 포함되지 않았는지 확인
"""

import os
import re
from pathlib import Path

def test_simulator_improvements():
    """시뮬레이터 개선 효과 테스트"""
    print("🔍 시뮬레이터 시나리오별 개선 효과 테스트...")
    
    # 시뮬레이터 파일 경로
    simulator_file = "web/bds_simulator.html"
    
    if not os.path.exists(simulator_file):
        print(f"❌ 시뮬레이터 파일을 찾을 수 없습니다: {simulator_file}")
        return False
    
    # 파일 내용 읽기
    with open(simulator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 예상되는 시나리오별 개선 효과 (정확한 값)
    expected_improvements = {
        '균형발전': 0.5,
        '경제중심': 0.4,
        '사회중심': 0.35,
        '환경중심': 0.3,
        '문화예술중심': 0.25,
        '안전중심': 0.3,
        '주거중심': 0.25,
        'R&D중심': 0.4,
        '혁신중심': 0.4
    }
    
    tests = {}
    all_passed = True
    
    print("\n📊 시나리오별 개선 효과 검증:")
    print("-" * 50)
    
    for scenario, expected_value in expected_improvements.items():
        # JavaScript 코드에서 해당 시나리오의 개선 효과가 올바르게 설정되어 있는지 확인
        pattern = rf"'{scenario}':\s*{expected_value}"
        is_correct = bool(re.search(pattern, content))
        tests[f"{scenario}_improvement"] = is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {scenario}: {expected_value} ({'통과' if is_correct else '실패'})")
        
        if not is_correct:
            all_passed = False
    
    # 잘못된 값(70.38%)이 포함되어 있지 않은지 확인
    print("\n🔍 잘못된 값 검사:")
    print("-" * 30)
    
    wrong_values = [
        "0.7037679110577804",
        "70.38",
        "70.38%",
        "0.7038"
    ]
    
    for wrong_value in wrong_values:
        if wrong_value in content:
            print(f"❌ 잘못된 값 발견: {wrong_value}")
            all_passed = False
        else:
            print(f"✅ 잘못된 값 없음: {wrong_value}")
    
    # 결과 요약
    print("\n" + "="*60)
    if all_passed:
        print("✅ 모든 시나리오의 개선 효과가 올바르게 설정되었습니다!")
        print("🚀 Push를 진행할 수 있습니다.")
        return True
    else:
        print("❌ 일부 시나리오의 개선 효과가 잘못되었습니다!")
        print("🔧 시뮬레이터 코드를 수정한 후 다시 테스트하세요.")
        return False

if __name__ == "__main__":
    success = test_simulator_improvements()
    exit(0 if success else 1)
