#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최소 GitHub Pages 테스트 (핵심만 검증)
- 파일 존재 확인
- 기본 HTML 구조 확인
- 중복 변수 선언만 검사
"""

import os
import sys
import re
import json
import time
from typing import Dict

class MinimalGitHubPagesTester:
    """최소 GitHub Pages 테스트 클래스"""
    
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
                "title_tag": "<title>" in content
            }
            
            passed = sum(essential_tags.values())
            total = len(essential_tags)
            
            return {
                "status": "PASS" if passed >= 4 else "FAIL",
                "score": f"{passed}/{total}",
                "details": essential_tags
            }
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def test_duplicate_vars(self, file_path: str) -> Dict:
        """중복 변수 선언 검사"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 중복 변수 선언 검사
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
    
    def test_file_size(self, file_path: str) -> Dict:
        """파일 크기 테스트"""
        try:
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            
            # 크기 기준 (매우 관대하게)
            if size_mb < 2.0:
                return {"status": "PASS", "size": f"{size_mb:.2f}MB", "message": "적당함"}
            else:
                return {"status": "WARN", "size": f"{size_mb:.2f}MB", "message": "큼"}
            
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("🚀 GitHub Pages 최소 테스트 시작")
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
            
            # 중복 변수 선언 테스트
            page_results["duplicate_vars"] = self.test_duplicate_vars(file_path)
            
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
        report.append("# GitHub Pages 최소 테스트 보고서")
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
                
                if "duplicate_vars" in result and result["duplicate_vars"]:
                    report.append("**중복 변수**:")
                    for var in result["duplicate_vars"]:
                        report.append(f"- {var}")
                
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
    print("🚀 GitHub Pages 최소 테스트 시작")
    print("="*60)
    
    tester = MinimalGitHubPagesTester()
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
