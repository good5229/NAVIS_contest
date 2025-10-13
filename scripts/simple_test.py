#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 GitHub Pages 테스트 (외부 의존성 최소화)
- HTML 파일 존재 확인
- JavaScript 구문 오류 검사
- 기본 구조 검증
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, List

class SimpleGitHubPagesTester:
    """간단한 GitHub Pages 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "dashboard": {},
            "simulator": {},
            "overall": {}
        }
        self.errors = []
        
    def test_file_exists(self, file_path: str) -> Dict:
        """파일 존재 확인"""
        if os.path.exists(file_path):
            return {"status": "PASS", "message": f"파일 존재: {file_path}"}
        else:
            return {"status": "FAIL", "message": f"파일 없음: {file_path}"}
    
    def test_html_structure(self, file_path: str) -> Dict:
        """HTML 구조 테스트"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 기본 HTML 태그 검사
            tests = {
                "doctype": "<!DOCTYPE html>" in content,
                "html_tag": "<html" in content,
                "head_tag": "<head>" in content,
                "body_tag": "<body>" in content,
                "title_tag": "<title>" in content,
                "meta_charset": 'charset="UTF-8"' in content,
                "meta_viewport": 'name="viewport"' in content
            }
            
            # 페이지별 특화 검사
            if "dashboard" in file_path:
                tests.update({
                    "bootstrap_css": 'bootstrap' in content,
                    "plotly_js": 'plotly' in content,
                    "nav_tabs": 'nav-tabs' in content,
                    "tab_content": 'tab-content' in content,
                    "overview_tab": 'id="overview"' in content,
                    "correlation_tab": 'id="correlation"' in content,
                    "granger_tab": 'id="granger"' in content,
                    "trends_tab": 'id="trends"' in content
                })
            elif "simulator" in file_path:
                tests.update({
                    "bootstrap_css": 'bootstrap' in content,
                    "plotly_js": 'plotly' in content,
                    "scenario_cards": content.count('scenario-card') >= 9,
                    "simulator_container": 'id="simulator-container"' in content,
                    "results_section": 'id="results-section"' in content
                })
            
            passed = sum(tests.values())
            total = len(tests)
            
            return {
                "status": "PASS" if passed == total else "FAIL",
                "score": f"{passed}/{total}",
                "details": tests,
                "failed_tests": [k for k, v in tests.items() if not v]
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_javascript_syntax(self, file_path: str) -> Dict:
        """JavaScript 구문 오류 테스트"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            js_errors = []
            js_warnings = []
            
            # 기본 구문 오류 검사 (더 정확한 패턴)
            syntax_checks = [
                ("unclosed_strings", r'"[^"]*$'),
                ("unclosed_brackets", r'\{[^}]*$'),
                ("duplicate_vars", r'const\s+(\w+).*const\s+\1'),
                ("console_errors", r'console\.error'),
                ("undefined_vars", r'undefined\s*[^=]')
            ]
            
            for check_name, pattern in syntax_checks:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    # 더 관대한 기준 적용
                    if check_name in ["unclosed_strings", "unclosed_brackets"] and len(matches) > 10:
                        js_errors.append(f"{check_name}: {len(matches)} occurrences")
                    elif check_name == "duplicate_vars":
                        js_errors.append(f"{check_name}: {len(matches)} occurrences")
                    else:
                        js_warnings.append(f"{check_name}: {len(matches)} occurrences")
            
            # 특화 검사
            if "dashboard" in file_path:
                # BDS 데이터 중복 선언 검사
                bds_data_count = len(re.findall(r'const\s+bdsData\s*=', content))
                if bds_data_count > 1:
                    js_errors.append(f"bdsData 중복 선언: {bds_data_count}회")
                
                # 필수 함수 존재 검사
                required_functions = [
                    'createOverviewDistributionChart',
                    'createCorrelationScatterChart', 
                    'createGrangerCausalityChart',
                    'createRegionalTrendsChart'
                ]
                
                for func in required_functions:
                    if func not in content:
                        js_errors.append(f"필수 함수 누락: {func}")
            
            elif "simulator" in file_path:
                # 시뮬레이터 클래스 검사
                if 'EnhancedBDSSimulatorV2' not in content:
                    js_errors.append("시뮬레이터 클래스 누락: EnhancedBDSSimulatorV2")
                
                # 필수 함수 존재 검사
                required_functions = [
                    'simulateBDS',
                    'displayBDSSimulationResults',
                    'createBDSImprovementChart',
                    'createPolicyEffectsChart'
                ]
                
                for func in required_functions:
                    if func not in content:
                        js_errors.append(f"필수 함수 누락: {func}")
            
            status = "PASS" if not js_errors else "FAIL"
            
            return {
                "status": status,
                "errors": js_errors,
                "warnings": js_warnings,
                "error_count": len(js_errors),
                "warning_count": len(js_warnings)
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_file_size(self, file_path: str) -> Dict:
        """파일 크기 테스트"""
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            
            # 크기 기준
            size_tests = {
                "under_1mb": size_mb < 1.0,
                "under_500kb": size_mb < 0.5,
                "under_2mb": size_mb < 2.0
            }
            
            passed = sum(size_tests.values())
            total = len(size_tests)
            
            return {
                "status": "PASS" if passed >= 1 else "WARN",
                "score": f"{passed}/{total}",
                "file_size_mb": f"{size_mb:.2f}",
                "details": size_tests
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("🚀 GitHub Pages 테스트 시작")
        print("="*60)
        
        # 테스트할 파일들
        files = {
            "dashboard": "bds_enhanced_dashboard.html",
            "simulator": "bds_simulator.html"
        }
        
        overall_status = "PASS"
        
        for page_name, file_path in files.items():
            print(f"\n📊 {page_name.title()} 테스트 시작")
            print("-" * 40)
            
            page_results = {}
            
            # 파일 존재 확인
            page_results["file_exists"] = self.test_file_exists(file_path)
            
            if page_results["file_exists"]["status"] == "FAIL":
                page_results["overall"] = {"status": "FAIL", "reason": "파일 없음"}
                overall_status = "FAIL"
                self.test_results[page_name] = page_results
                continue
            
            # HTML 구조 테스트
            page_results["html_structure"] = self.test_html_structure(file_path)
            
            # JavaScript 구문 테스트
            page_results["javascript_syntax"] = self.test_javascript_syntax(file_path)
            
            # 파일 크기 테스트
            page_results["file_size"] = self.test_file_size(file_path)
            
            # 페이지별 결과 요약
            page_statuses = [result["status"] for result in page_results.values() if "status" in result]
            page_status = "PASS" if all(status in ["PASS", "WARN"] for status in page_statuses) else "FAIL"
            
            if page_status == "FAIL":
                overall_status = "FAIL"
            
            page_results["overall"] = {
                "status": page_status,
                "tests_passed": len([s for s in page_statuses if s == "PASS"]),
                "tests_total": len(page_statuses)
            }
            
            self.test_results[page_name] = page_results
            
            print(f"✅ {page_name.title()} 테스트 완료: {page_status}")
        
        # 전체 결과
        self.test_results["overall"] = {
            "status": overall_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": len(files),
            "passed_pages": len([p for p in self.test_results.values() if isinstance(p, dict) and p.get("overall", {}).get("status") == "PASS"])
        }
        
        return self.test_results
    
    def generate_report(self) -> str:
        """테스트 보고서 생성"""
        import time
        
        report = []
        report.append("# GitHub Pages 테스트 보고서")
        report.append(f"**생성 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**전체 상태**: {self.test_results['overall']['status']}")
        report.append("")
        
        for page_name, results in self.test_results.items():
            if page_name == "overall":
                continue
                
            report.append(f"## {page_name.title()} 페이지")
            report.append("")
            
            for test_name, result in results.items():
                if test_name == "overall":
                    continue
                    
                status_emoji = "✅" if result["status"] == "PASS" else "⚠️" if result["status"] == "WARN" else "❌"
                report.append(f"### {test_name.replace('_', ' ').title()}")
                report.append(f"**상태**: {status_emoji} {result['status']}")
                
                if "score" in result:
                    report.append(f"**점수**: {result['score']}")
                
                if "error" in result:
                    report.append(f"**오류**: {result['error']}")
                
                if "errors" in result and result["errors"]:
                    report.append("**오류 목록**:")
                    for error in result["errors"]:
                        report.append(f"- {error}")
                
                if "warnings" in result and result["warnings"]:
                    report.append("**경고 목록**:")
                    for warning in result["warnings"]:
                        report.append(f"- {warning}")
                
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self):
        """테스트 결과 저장"""
        import time
        
        # JSON 결과 저장
        with open('github_pages_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        # 보고서 저장
        report = self.generate_report()
        with open('github_pages_test_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n📁 테스트 결과 저장:")
        print("  • github_pages_test_results.json")
        print("  • github_pages_test_report.md")

def main():
    """메인 실행 함수"""
    print("🚀 GitHub Pages 테스트 시작")
    print("="*60)
    
    tester = SimpleGitHubPagesTester()
    results = tester.run_all_tests()
    
    # 결과 저장
    tester.save_results()
    
    # 최종 결과 출력
    print("\n📋 테스트 결과 요약")
    print("="*60)
    print(f"전체 상태: {results['overall']['status']}")
    print(f"테스트된 페이지: {results['overall']['total_pages']}개")
    print(f"통과한 페이지: {results['overall']['passed_pages']}개")
    
    # 실패한 테스트가 있으면 상세 정보 출력
    if results['overall']['status'] == "FAIL":
        print("\n❌ 실패한 테스트:")
        for page_name, page_results in results.items():
            if page_name == "overall":
                continue
            for test_name, result in page_results.items():
                if test_name == "overall":
                    continue
                if result["status"] == "FAIL":
                    print(f"  • {page_name}: {test_name}")
    
    # 종료 코드 설정
    if results['overall']['status'] == "FAIL":
        print("\n❌ 테스트 실패 - Push를 중단합니다.")
        sys.exit(1)
    else:
        print("\n✅ 모든 테스트 통과 - Push를 진행할 수 있습니다.")
        sys.exit(0)

if __name__ == "__main__":
    main()
