#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Pages 대시보드 및 시뮬레이션 페이지 테스트케이스
- HTML 구조 검증
- JavaScript 오류 검증
- 기능 테스트
- 성능 검증
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class GitHubPagesTester:
    """GitHub Pages 테스트 클래스"""
    
    def __init__(self):
        self.base_url = "https://good5229.github.io/NAVIS_contest"
        self.test_results = {
            "dashboard": {},
            "simulator": {},
            "overall": {}
        }
        self.errors = []
        self.warnings = []
        
    def test_html_structure(self, url: str, page_name: str) -> Dict:
        """HTML 구조 테스트"""
        print(f"🔍 {page_name} HTML 구조 테스트...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"status": "FAIL", "error": f"HTTP {response.status_code}"}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 기본 HTML 구조 검증
            tests = {
                "doctype": soup.find('!DOCTYPE') is not None or 'DOCTYPE' in str(soup),
                "html_tag": soup.find('html') is not None,
                "head_tag": soup.find('head') is not None,
                "body_tag": soup.find('body') is not None,
                "title_tag": soup.find('title') is not None,
                "meta_charset": soup.find('meta', {'charset': True}) is not None,
                "meta_viewport": soup.find('meta', {'name': 'viewport'}) is not None
            }
            
            # 페이지별 특화 테스트
            if page_name == "Dashboard":
                tests.update({
                    "bootstrap_css": 'bootstrap' in str(soup.find('link', {'href': True})),
                    "plotly_js": 'plotly' in str(soup.find('script', {'src': True})),
                    "nav_tabs": soup.find('ul', {'class': 'nav-tabs'}) is not None,
                    "tab_content": soup.find('div', {'class': 'tab-content'}) is not None,
                    "overview_tab": soup.find('div', {'id': 'overview'}) is not None,
                    "correlation_tab": soup.find('div', {'id': 'correlation'}) is not None,
                    "granger_tab": soup.find('div', {'id': 'granger'}) is not None,
                    "trends_tab": soup.find('div', {'id': 'trends'}) is not None
                })
            elif page_name == "Simulator":
                tests.update({
                    "bootstrap_css": 'bootstrap' in str(soup.find('link', {'href': True})),
                    "plotly_js": 'plotly' in str(soup.find('script', {'src': True})),
                    "scenario_cards": len(soup.find_all('div', {'class': 'scenario-card'})) >= 9,
                    "simulator_container": soup.find('div', {'id': 'simulator-container'}) is not None,
                    "results_section": soup.find('div', {'id': 'results-section'}) is not None
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
    
    def test_javascript_syntax(self, url: str, page_name: str) -> Dict:
        """JavaScript 구문 오류 테스트"""
        print(f"🔍 {page_name} JavaScript 구문 테스트...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"status": "FAIL", "error": f"HTTP {response.status_code}"}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            scripts = soup.find_all('script')
            
            js_errors = []
            js_warnings = []
            
            for script in scripts:
                if script.string:
                    js_code = script.string
                    
                    # 기본 구문 오류 검사
                    syntax_checks = [
                        ("unclosed_strings", r'"[^"]*$|\'[^\']*$'),
                        ("unclosed_brackets", r'\{[^}]*$|\[[^\]]*$'),
                        ("duplicate_vars", r'const\s+(\w+).*const\s+\1'),
                        ("missing_semicolons", r'}\s*$'),
                        ("console_errors", r'console\.error'),
                        ("undefined_vars", r'undefined\s*[^=]')
                    ]
                    
                    for check_name, pattern in syntax_checks:
                        matches = re.findall(pattern, js_code, re.MULTILINE)
                        if matches:
                            if check_name in ["unclosed_strings", "unclosed_brackets", "duplicate_vars"]:
                                js_errors.append(f"{check_name}: {len(matches)} occurrences")
                            else:
                                js_warnings.append(f"{check_name}: {len(matches)} occurrences")
            
            # 특화 검사
            if page_name == "Dashboard":
                # BDS 데이터 중복 선언 검사
                bds_data_count = len(re.findall(r'const\s+bdsData\s*=', response.text))
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
                    if func not in response.text:
                        js_errors.append(f"필수 함수 누락: {func}")
            
            elif page_name == "Simulator":
                # 시뮬레이터 클래스 검사
                if 'EnhancedBDSSimulatorV2' not in response.text:
                    js_errors.append("시뮬레이터 클래스 누락: EnhancedBDSSimulatorV2")
                
                # 필수 함수 존재 검사
                required_functions = [
                    'simulateBDS',
                    'displayBDSSimulationResults',
                    'createBDSImprovementChart',
                    'createPolicyEffectsChart'
                ]
                
                for func in required_functions:
                    if func not in response.text:
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
    
    def test_responsive_design(self, url: str, page_name: str) -> Dict:
        """반응형 디자인 테스트"""
        print(f"🔍 {page_name} 반응형 디자인 테스트...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"status": "FAIL", "error": f"HTTP {response.status_code}"}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Bootstrap 클래스 검사
            bootstrap_classes = [
                'container', 'row', 'col-', 'btn', 'card', 'modal',
                'nav-tabs', 'tab-content', 'table', 'alert'
            ]
            
            found_classes = []
            for class_name in bootstrap_classes:
                if soup.find(class_=re.compile(class_name)):
                    found_classes.append(class_name)
            
            # 반응형 클래스 검사
            responsive_classes = ['col-md-', 'col-lg-', 'col-sm-', 'col-xs-']
            responsive_found = any(
                soup.find(class_=re.compile(cls)) for cls in responsive_classes
            )
            
            # 메타 뷰포트 검사
            viewport_meta = soup.find('meta', {'name': 'viewport'})
            viewport_correct = viewport_meta and 'width=device-width' in str(viewport_meta)
            
            tests = {
                "bootstrap_classes": len(found_classes) >= 5,
                "responsive_classes": responsive_found,
                "viewport_meta": viewport_correct,
                "mobile_friendly": viewport_correct and responsive_found
            }
            
            passed = sum(tests.values())
            total = len(tests)
            
            return {
                "status": "PASS" if passed >= 3 else "FAIL",
                "score": f"{passed}/{total}",
                "details": tests,
                "bootstrap_classes_found": found_classes
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_performance(self, url: str, page_name: str) -> Dict:
        """성능 테스트"""
        print(f"🔍 {page_name} 성능 테스트...")
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=30)
            load_time = time.time() - start_time
            
            if response.status_code != 200:
                return {"status": "FAIL", "error": f"HTTP {response.status_code}"}
            
            # 파일 크기 검사
            content_size = len(response.content)
            size_mb = content_size / (1024 * 1024)
            
            # 성능 기준
            performance_tests = {
                "load_time_under_5s": load_time < 5.0,
                "load_time_under_3s": load_time < 3.0,
                "size_under_1mb": size_mb < 1.0,
                "size_under_500kb": size_mb < 0.5
            }
            
            passed = sum(performance_tests.values())
            total = len(performance_tests)
            
            return {
                "status": "PASS" if passed >= 2 else "WARN",
                "score": f"{passed}/{total}",
                "load_time": f"{load_time:.2f}s",
                "file_size": f"{size_mb:.2f}MB",
                "details": performance_tests
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_accessibility(self, url: str, page_name: str) -> Dict:
        """접근성 테스트"""
        print(f"🔍 {page_name} 접근성 테스트...")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"status": "FAIL", "error": f"HTTP {response.status_code}"}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 접근성 검사
            accessibility_tests = {
                "alt_texts": len(soup.find_all('img', alt=True)) > 0,
                "heading_structure": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])) > 0,
                "form_labels": len(soup.find_all('label')) > 0,
                "aria_labels": len(soup.find_all(attrs={'aria-label': True})) > 0,
                "lang_attribute": soup.find('html', {'lang': True}) is not None,
                "contrast_colors": 'color' in str(soup.find('style')) or 'background' in str(soup.find('style'))
            }
            
            passed = sum(accessibility_tests.values())
            total = len(accessibility_tests)
            
            return {
                "status": "PASS" if passed >= 3 else "WARN",
                "score": f"{passed}/{total}",
                "details": accessibility_tests
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("🚀 GitHub Pages 테스트 시작")
        print("="*60)
        
        # 테스트할 페이지들
        pages = {
            "Dashboard": f"{self.base_url}/bds_enhanced_dashboard.html",
            "Simulator": f"{self.base_url}/bds_simulator.html"
        }
        
        overall_status = "PASS"
        
        for page_name, url in pages.items():
            print(f"\n📊 {page_name} 테스트 시작")
            print("-" * 40)
            
            page_results = {}
            
            # HTML 구조 테스트
            page_results["html_structure"] = self.test_html_structure(url, page_name)
            
            # JavaScript 구문 테스트
            page_results["javascript_syntax"] = self.test_javascript_syntax(url, page_name)
            
            # 반응형 디자인 테스트
            page_results["responsive_design"] = self.test_responsive_design(url, page_name)
            
            # 성능 테스트
            page_results["performance"] = self.test_performance(url, page_name)
            
            # 접근성 테스트
            page_results["accessibility"] = self.test_accessibility(url, page_name)
            
            # 페이지별 결과 요약
            page_statuses = [result["status"] for result in page_results.values()]
            page_status = "PASS" if all(status in ["PASS", "WARN"] for status in page_statuses) else "FAIL"
            
            if page_status == "FAIL":
                overall_status = "FAIL"
            
            page_results["overall"] = {
                "status": page_status,
                "tests_passed": len([s for s in page_statuses if s == "PASS"]),
                "tests_total": len(page_statuses)
            }
            
            self.test_results[page_name.lower().replace(" ", "_")] = page_results
            
            print(f"✅ {page_name} 테스트 완료: {page_status}")
        
        # 전체 결과
        self.test_results["overall"] = {
            "status": overall_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": len(pages),
            "passed_pages": len([p for p in self.test_results.values() if isinstance(p, dict) and p.get("overall", {}).get("status") == "PASS"])
        }
        
        return self.test_results
    
    def generate_report(self) -> str:
        """테스트 보고서 생성"""
        report = []
        report.append("# GitHub Pages 테스트 보고서")
        report.append(f"**생성 시간**: {self.test_results['overall']['timestamp']}")
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
    
    tester = GitHubPagesTester()
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
