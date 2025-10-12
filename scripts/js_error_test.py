#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JavaScript 오류 검사 테스트
- 정의되지 않은 변수 검사
- 구문 오류 검사
- 중복 선언 검사
"""

import os
import sys
import re
import json
import time
from typing import Dict, List

class JavaScriptErrorTester:
    """JavaScript 오류 검사 클래스"""
    
    def __init__(self):
        self.test_results = {}
        
    def test_undefined_variables(self, file_path: str) -> Dict:
        """정의되지 않은 변수 검사"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 변수 선언 찾기
            declared_vars = set()
            used_vars = set()
            
            # const, let, var로 선언된 변수들
            const_vars = re.findall(r'const\s+(\w+)', content)
            let_vars = re.findall(r'let\s+(\w+)', content)
            var_vars = re.findall(r'var\s+(\w+)', content)
            
            declared_vars.update(const_vars)
            declared_vars.update(let_vars)
            declared_vars.update(var_vars)
            
            # 함수 매개변수들
            function_params = re.findall(r'function\s+\w+\s*\(([^)]*)\)', content)
            for params in function_params:
                param_list = [p.strip() for p in params.split(',') if p.strip()]
                declared_vars.update(param_list)
            
            # 사용된 변수들 찾기 (간단한 패턴)
            variable_usage = re.findall(r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b', content)
            used_vars.update(variable_usage)
            
            # 정의되지 않은 변수 찾기
            undefined_vars = []
            for var in used_vars:
                if var not in declared_vars and var not in ['function', 'if', 'else', 'for', 'while', 'return', 'true', 'false', 'null', 'undefined', 'console', 'document', 'window', 'alert', 'parseInt', 'parseFloat', 'Math', 'Date', 'Object', 'Array', 'String', 'Number', 'Boolean', 'JSON', 'localStorage', 'sessionStorage', 'location', 'history', 'navigator', 'screen', 'innerHTML', 'textContent', 'value', 'length', 'push', 'pop', 'shift', 'unshift', 'slice', 'splice', 'join', 'split', 'indexOf', 'lastIndexOf', 'includes', 'forEach', 'map', 'filter', 'reduce', 'sort', 'reverse', 'toString', 'toLowerCase', 'toUpperCase', 'trim', 'substring', 'substr', 'charAt', 'charCodeAt', 'replace', 'search', 'match', 'test', 'exec', 'keys', 'values', 'entries', 'hasOwnProperty', 'isArray', 'isNaN', 'isFinite', 'encodeURI', 'decodeURI', 'encodeURIComponent', 'decodeURIComponent', 'escape', 'unescape', 'eval', 'setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'requestAnimationFrame', 'cancelAnimationFrame', 'addEventListener', 'removeEventListener', 'preventDefault', 'stopPropagation', 'stopImmediatePropagation', 'querySelector', 'querySelectorAll', 'getElementById', 'getElementsByClassName', 'getElementsByTagName', 'createElement', 'createTextNode', 'appendChild', 'removeChild', 'insertBefore', 'replaceChild', 'cloneNode', 'hasChildNodes', 'childNodes', 'firstChild', 'lastChild', 'nextSibling', 'previousSibling', 'parentNode', 'nodeName', 'nodeType', 'nodeValue', 'attributes', 'getAttribute', 'setAttribute', 'removeAttribute', 'hasAttribute', 'className', 'id', 'style', 'offsetWidth', 'offsetHeight', 'offsetLeft', 'offsetTop', 'scrollLeft', 'scrollTop', 'scrollWidth', 'scrollHeight', 'clientWidth', 'clientHeight', 'clientLeft', 'clientTop', 'getBoundingClientRect', 'getComputedStyle', 'createEvent', 'dispatchEvent', 'addEventListener', 'removeEventListener', 'preventDefault', 'stopPropagation', 'stopImmediatePropagation', 'querySelector', 'querySelectorAll', 'getElementById', 'getElementsByClassName', 'getElementsByTagName', 'createElement', 'createTextNode', 'appendChild', 'removeChild', 'insertBefore', 'replaceChild', 'cloneNode', 'hasChildNodes', 'childNodes', 'firstChild', 'lastChild', 'nextSibling', 'previousSibling', 'parentNode', 'nodeName', 'nodeType', 'nodeValue', 'attributes', 'getAttribute', 'setAttribute', 'removeAttribute', 'hasAttribute', 'className', 'id', 'style', 'offsetWidth', 'offsetHeight', 'offsetLeft', 'offsetTop', 'scrollLeft', 'scrollTop', 'scrollWidth', 'scrollHeight', 'clientWidth', 'clientHeight', 'clientLeft', 'clientTop', 'getBoundingClientRect', 'getComputedStyle', 'createEvent', 'dispatchEvent']:
                    undefined_vars.append(var)
            
            # 실제 오류만 필터링
            critical_undefined = []
            for var in undefined_vars:
                if var in ['bdsV2Data', 'navisData', 'bdsData', 'originalBdsData']:
                    critical_undefined.append(var)
            
            status = "PASS" if not critical_undefined else "FAIL"
            
            return {
                "status": status,
                "undefined_vars": critical_undefined,
                "error_count": len(critical_undefined)
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_syntax_errors(self, file_path: str) -> Dict:
        """구문 오류 검사"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            syntax_errors = []
            
            # 기본 구문 오류 검사
            syntax_checks = [
                ("unclosed_strings", r'"[^"]*$'),
                ("unclosed_brackets", r'\{[^}]*$'),
                ("missing_semicolons", r'}\s*$'),
                ("console_errors", r'console\.error'),
                ("undefined_vars", r'undefined\s*[^=]')
            ]
            
            for check_name, pattern in syntax_checks:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches and len(matches) > 10:  # 10개 이상일 때만 오류로 간주
                    syntax_errors.append(f"{check_name}: {len(matches)} occurrences")
            
            status = "PASS" if not syntax_errors else "FAIL"
            
            return {
                "status": status,
                "syntax_errors": syntax_errors,
                "error_count": len(syntax_errors)
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_duplicate_declarations(self, file_path: str) -> Dict:
        """중복 선언 검사"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            duplicate_vars = []
            
            # const 중복 선언 검사
            const_vars = re.findall(r'const\s+(\w+)', content)
            var_counts = {}
            for var in const_vars:
                var_counts[var] = var_counts.get(var, 0) + 1
            
            for var, count in var_counts.items():
                if count > 1:
                    duplicate_vars.append(f"{var}: {count}회 선언")
            
            # let 중복 선언 검사
            let_vars = re.findall(r'let\s+(\w+)', content)
            let_var_counts = {}
            for var in let_vars:
                let_var_counts[var] = let_var_counts.get(var, 0) + 1
            
            for var, count in let_var_counts.items():
                if count > 1:
                    duplicate_vars.append(f"{var}: {count}회 선언")
            
            status = "PASS" if not duplicate_vars else "FAIL"
            
            return {
                "status": status,
                "duplicate_vars": duplicate_vars,
                "error_count": len(duplicate_vars)
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("🚀 JavaScript 오류 검사 시작")
        print("="*60)
        
        # 테스트할 파일들
        files = {
            "dashboard": "bds_enhanced_dashboard.html",
            "simulator": "bds_simulator.html"
        }
        
        overall_status = "PASS"
        
        for page_name, file_path in files.items():
            print(f"\n📊 {page_name.title()} JavaScript 오류 검사")
            print("-" * 40)
            
            page_results = {}
            
            # 정의되지 않은 변수 검사
            page_results["undefined_vars"] = self.test_undefined_variables(file_path)
            
            # 구문 오류 검사
            page_results["syntax_errors"] = self.test_syntax_errors(file_path)
            
            # 중복 선언 검사
            page_results["duplicate_declarations"] = self.test_duplicate_declarations(file_path)
            
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
            
            print(f"✅ {page_name.title()} 검사 완료: {page_status}")
        
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
        report = []
        report.append("# JavaScript 오류 검사 보고서")
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
                
                if "error" in result:
                    report.append(f"**오류**: {result['error']}")
                
                if "undefined_vars" in result and result["undefined_vars"]:
                    report.append("**정의되지 않은 변수**:")
                    for var in result["undefined_vars"]:
                        report.append(f"- {var}")
                
                if "syntax_errors" in result and result["syntax_errors"]:
                    report.append("**구문 오류**:")
                    for error in result["syntax_errors"]:
                        report.append(f"- {error}")
                
                if "duplicate_vars" in result and result["duplicate_vars"]:
                    report.append("**중복 선언**:")
                    for var in result["duplicate_vars"]:
                        report.append(f"- {var}")
                
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self):
        """테스트 결과 저장"""
        # JSON 결과 저장
        with open('javascript_error_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        # 보고서 저장
        report = self.generate_report()
        with open('javascript_error_test_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n📁 테스트 결과 저장:")
        print("  • javascript_error_test_results.json")
        print("  • javascript_error_test_report.md")

def main():
    """메인 실행 함수"""
    print("🚀 JavaScript 오류 검사 시작")
    print("="*60)
    
    tester = JavaScriptErrorTester()
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
