#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDS 분석 실행 스크립트
- BDS 계산 및 시각화
- NAVIS 지역발전지수와의 비교
- 지역 균형 개선 시뮬레이션
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

def check_requirements():
    """필요한 파일들이 존재하는지 확인"""
    required_files = [
        "bds_analysis.py",
        "navis_analysis.py",
        "skorea-provinces-2018-geo.json",
        "navis_data/1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 다음 파일들이 누락되었습니다:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # KOSIS API 키 확인
    if not os.getenv("KOSIS_API_KEY"):
        print("❌ KOSIS_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 KOSIS_API_KEY=your_api_key를 추가하세요.")
        return False
    
    print("✅ 모든 요구사항이 충족되었습니다.")
    return True

def run_bds_analysis():
    """BDS 분석 실행"""
    print("\n=== 1단계: BDS 분석 실행 ===")
    
    try:
        from bds_analysis import build_timeseries_and_map
        
        print("BDS 계산 및 시각화를 시작합니다...")
        bds_data = build_timeseries_and_map(
            weight_mode="pca",
            min_year=2000,
            geojson_path="./skorea-provinces-2018-geo.json",
            out_csv="bds_timeseries.csv",
            out_html="bds_choropleth.html"
        )
        
        print("✅ BDS 분석이 완료되었습니다.")
        return bds_data
        
    except Exception as e:
        print(f"❌ BDS 분석 중 오류 발생: {e}")
        return None

def run_navis_analysis():
    """NAVIS 분석 실행"""
    print("\n=== 2단계: NAVIS 분석 실행 ===")
    
    try:
        from navis_analysis import main as navis_main
        
        print("NAVIS 지역발전지수 분석을 시작합니다...")
        navis_main()
        
        print("✅ NAVIS 분석이 완료되었습니다.")
        return True
        
    except Exception as e:
        print(f"❌ NAVIS 분석 중 오류 발생: {e}")
        return False

def show_results():
    """결과 파일들 표시"""
    print("\n=== 생성된 결과 파일들 ===")
    
    output_dir = Path("outputs_timeseries")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            for file_path in sorted(files):
                size = file_path.stat().st_size
                print(f"📄 {file_path.name} ({size:,} bytes)")
        else:
            print("❌ 결과 파일이 생성되지 않았습니다.")
    else:
        print("❌ outputs_timeseries 디렉토리가 존재하지 않습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 BDS 지역 균형발전 지수 분석 시스템")
    print("=" * 50)
    
    # 1. 요구사항 확인
    if not check_requirements():
        print("\n❌ 분석을 시작할 수 없습니다. 위의 문제들을 해결해주세요.")
        return
    
    # 2. BDS 분석 실행
    bds_data = run_bds_analysis()
    if bds_data is None:
        print("\n❌ BDS 분석이 실패했습니다.")
        return
    
    # 3. NAVIS 분석 실행
    navis_success = run_navis_analysis()
    if not navis_success:
        print("\n⚠ NAVIS 분석이 실패했습니다. BDS 분석 결과는 사용 가능합니다.")
    
    # 4. 결과 표시
    show_results()
    
    print("\n🎉 분석이 완료되었습니다!")
    print("\n📊 주요 결과 파일:")
    print("   - bds_choropleth.html: 인터랙티브 지도 시각화")
    print("   - bds_navis_comparison.html: NAVIS 지수와의 비교")
    print("   - regional_balance_simulation.html: 지역 균형 개선 시뮬레이션")
    print("   - analysis_report.md: 종합 분석 보고서")
    
    print("\n💡 사용 팁:")
    print("   - HTML 파일들을 웹 브라우저에서 열어보세요")
    print("   - analysis_report.md를 마크다운 뷰어로 확인하세요")
    print("   - 추가 분석이 필요하면 bds_analysis.py나 navis_analysis.py를 수정하세요")

if __name__ == "__main__":
    main() 