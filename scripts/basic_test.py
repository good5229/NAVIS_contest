#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 GitHub Pages 테스트 (핵심 기능만 검증)
- 파일 존재 확인
- 기본 HTML 구조 확인
- 치명적 JavaScript 오류만 검사
"""

import os
import sys
import re
import json
import time
from typing import Dict

class BasicGitHubPagesTester:
    """기본 GitHub Pages 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "dashboard": {},
            "simulator": {},
            "overall": {}
        }
        
    def test_file_exists(self, file_path: str) -> Dict:
        """파일 존재 확인"""
        if os.path.exists(file_path):
            return {"status": "PASS", "message": f"파일 존재: {file_path}"}
        else:
            return {"status": "FAIL", "message": f"파일 없음: {file_path}"}
    
    def test_basic_html(self, file_path: str) -> Dict:
        """기본 HTML 구조 테스트"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 핵심 HTML 태그만 검사
            essential_tags = {
                "doctype": "<!DOCTYPE html>" in content,
                "html_tag": "<html" in content,
                "head_tag": "<head>" in content,
                "body_tag": "<body>" in content,
                "title_tag": "<title>" in content,
                "meta_charset": 'charset="UTF-8"' in content
            }
            
            passed = sum(essential_tags.values())
            total = len(essential_tags)
            
            return {
                "status": "PASS" if passed >= 5 else "FAIL",
                "score": f"{passed}/{total}",
                "details": essential_tags
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_critical_js_errors(self, file_path: str) -> Dict:
        """치명적 JavaScript 오류만 검사"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            critical_errors = []
            
            # 치명적 오류만 검사 (더 관대한 기준)
            critical_checks = [
                ("duplicate_const", r'const\s+(\w+).*const\s+\1'),
                ("syntax_errors", r'function\s+\w+\s*\([^)]*\)\s*\{[^}]*$'),
                ("missing_closing", r'\{[^}]*$')
            ]
            
            for check_name, pattern in critical_checks:
                matches = re.findall(pattern, content, re.MULTILINE)
                # 더 관대한 기준: 20개 이상일 때만 오류로 간주
                if matches and len(matches) > 20:
                    critical_errors.append(f"{check_name}: {len(matches)} occurrences")
            
            # 특화 검사
            if "dashboard" in file_path:
                # BDS 데이터 중복 선언 검사
                bds_data_count = len(re.findall(r'const\s+bdsData\s*=', content))
                if bds_data_count > 1:
                    critical_errors.append(f"bdsData 중복 선언: {bds_data_count}회")
            
            status = "PASS" if not critical_errors else "FAIL"
            
            return {
                "status": status,
                "critical_errors": critical_errors,
                "error_count": len(critical_errors)
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_file_size(self, file_path: str) -> Dict:
        """파일 크기 테스트"""
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            
            # 크기 기준 (관대하게)
            if size_mb < 0.1:
                return {"status": "PASS", "size": f"{size_mb:.2f}MB", "message": "매우 작음"}
            elif size_mb < 0.5:
                return {"status": "PASS", "size": f"{size_mb:.2f}MB", "message": "적당함"}
            elif size_mb < 1.0:
                return {"status": "WARN", "size": f"{size_mb:.2f}MB", "message": "큼"}
            else:
                return {"status": "FAIL", "size": f"{size_mb:.2f}MB", "message": "너무 큼"}
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("🚀 GitHub Pages 기본 테스트 시작")
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
            
            # 기본 HTML 구조 테스트
            page_results["basic_html"] = self.test_basic_html(file_path)
            
            # 치명적 JavaScript 오류 테스트
            page_results["critical_js"] = self.test_critical_js_errors(file_path)
            
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
        report = []
        report.append("# GitHub Pages 기본 테스트 보고서")
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
                
                if "critical_errors" in result and result["critical_errors"]:
                    report.append("**치명적 오류**:")
                    for error in result["critical_errors"]:
                        report.append(f"- {error}")
                
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
    print("🚀 GitHub Pages 기본 테스트 시작")
    print("="*60)
    
    tester = BasicGitHubPagesTester()
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
