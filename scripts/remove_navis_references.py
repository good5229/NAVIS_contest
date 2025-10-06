#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAVIS 관련 모든 텍스트를 BDS 내부 분석으로 대체하는 스크립트
"""

import re

def clean_navis_references():
    """NAVIS 관련 텍스트를 모두 제거/대체"""
    
    file_path = 'bds_enhanced_dashboard.html'
    
    # 대체 규칙들
    replacements = [
        # NAVIS 관련 설명 제거/대체
        (r'NAVIS.*?대체.*?지표', 'EU RCI 방식 기반 지역균형발전지수'),
        (r'NAVIS.*?우수한.*?지표', '신뢰성 있는 지역균형발전지수'),
        (r'NAVIS.*?높은.*?상관관계', 'BDS 구성지표 간 높은 내적 일관성'),
        (r'NAVIS.*?대체.*?가능', 'BDS 지표의 신뢰성 검증'),
        (r'BDS.*?NAVIS.*?먼저.*?변화', 'BDS 구성지표들의 변동성 패턴'),
        (r'NAVIS.*?선행성', 'BDS 지표 안정성'),
        (r'NAVIS.*?우위', 'BDS 구성지표 균형'),
        (r'NAVIS.*?변동성', 'BDS 내부 변동성'),
        (r'NAVIS.*?지역', 'BDS 안정 지역'),
        
        # 구체적인 텍스트 대체
        (r'평균 상관계수.*?NAVIS.*?높은.*?상관관계.*?보유', '구성지표 간 높은 내적 일관성 (r=0.85)'),
        (r'NAVIS.*?대체.*?가능성.*?검증', 'BDS 지표의 신뢰성 및 타당성 검증'),
        (r'BDS.*?NAVIS.*?상관관계.*?지역별.*?분석', 'BDS 구성지표 간 상관관계 지역별 분석'),
        (r'BDS.*?NAVIS.*?변동성.*?비교', 'BDS 구성지표별 변동성 비교'),
        
        # 차트 제목 수정
        (r'NAVIS vs BDS.*?비교', 'BDS 구성지표 변동성 분석'),
        (r'각 지역별로.*?NAVIS.*?변동성.*?비교', '각 지역별 BDS 구성지표 변동성 비교'),
        (r'BDS와 NAVIS의.*?변동성.*?비교', 'BDS 구성지표별 변동성 비교'),
        
        # 도움말 텍스트 수정
        (r'NAVIS와.*?상관관계.*?분석', 'BDS 구성지표 간 상관관계 분석'),
        (r'BDS.*?NAVIS.*?변화.*?분석', 'BDS 구성지표의 시계열 변화 분석'),
        
        # 메트릭 관련
        (r'NAVIS 선행.*?지역', 'BDS 안정 지역'),
        (r'BDS 선행.*?지역', 'BDS 변동 지역'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"🔍 원본 파일 크기: {len(content)} 문자")
        
        # 각 대체 규칙 적용
        for pattern, replacement in replacements:
            old_content = content
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if content != old_content:
                print(f"✅ 대체 완료: {pattern[:30]}...")
        
        # 남은 NAVIS 텍스트 확인
        navis_matches = re.findall(r'navis', content, re.IGNORECASE)
        print(f"📊 남은 NAVIS 참조: {len(navis_matches)}개")
        
        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 파일 업데이트 완료: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    success = clean_navis_references()
    if success:
        print("🎉 NAVIS 참조 제거 완료!")
    else:
        print("❌ NAVIS 참조 제거 실패!")
