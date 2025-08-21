#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통계청 KOSIS OPEN API 데이터 수집기

목표: 지역균형발전과 관련된 다양한 실험적 지표들을 수집하여
BDS 모델의 독립성과 선행성을 강화

수집 지표:
1. 경제지표 (GDP, 고용률, 소득 등)
2. 사회지표 (인구, 교육, 의료 등)
3. 환경지표 (대기질, 녹지율 등)
4. 인프라지표 (교통, 통신, 에너지 등)
5. 혁신지표 (특허, R&D, 창업 등)
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# 환경변수 로드
load_dotenv()

class KosisDataCollector:
    def __init__(self):
        """KOSIS API 초기화"""
        self.api_key = os.getenv('KOSIS_OPEN_API')
        if not self.api_key:
            raise ValueError("KOSIS_OPEN_API 환경변수가 설정되지 않았습니다.")
        
        self.base_url = "https://kosis.kr/openapi/statisticsData.do"
        self.regions = {
            '서울특별시': '11000', '부산광역시': '21000', '대구광역시': '22000',
            '인천광역시': '23000', '광주광역시': '24000', '대전광역시': '25000',
            '울산광역시': '26000', '세종특별자치시': '41000',
            '경기도': '31000', '강원도': '32000', '충청북도': '33000',
            '충청남도': '34000', '전라북도': '35000', '전라남도': '36000',
            '경상북도': '37000', '경상남도': '38000', '제주특별자치도': '39000'
        }
        
        print("✅ KOSIS 데이터 수집기 초기화 완료")
        print(f"📊 수집 대상 지역: {len(self.regions)}개")
    
    def get_kosis_data(self, tblId, prdSe, startPrdDe, endPrdDe, orgId, prdSe2=None, prdSe3=None):
        """KOSIS API 데이터 수집"""
        params = {
            'method': 'getList',
            'apiKey': self.api_key,
            'format': 'json',
            'jsonVD': 'Y',
            'userFields': '',
            'prdSe': prdSe,
            'startPrdDe': startPrdDe,
            'endPrdDe': endPrdDe,
            'orgId': orgId,
            'tblId': tblId
        }
        
        if prdSe2:
            params['prdSe2'] = prdSe2
        if prdSe3:
            params['prdSe3'] = prdSe3
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'ErrMsg' in data:
                print(f"❌ API 오류: {data['ErrMsg']}")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None
    
    def collect_economic_indicators(self):
        """경제지표 수집"""
        print("\n=== 경제지표 수집 ===")
        
        economic_data = {}
        
        # 1. 지역내총생산(GRDP) - 실질성장률
        print("1. 지역내총생산(GRDP) 실질성장률 수집 중...")
        grdp_data = self.get_kosis_data(
            tblId='DT_1C0001', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if grdp_data:
            economic_data['grdp_growth'] = self.process_kosis_data(grdp_data, 'GRDP 실질성장률')
        
        # 2. 고용률
        print("2. 고용률 수집 중...")
        employment_data = self.get_kosis_data(
            tblId='DT_1DA7002', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if employment_data:
            economic_data['employment_rate'] = self.process_kosis_data(employment_data, '고용률')
        
        # 3. 1인당 개인소득
        print("3. 1인당 개인소득 수집 중...")
        income_data = self.get_kosis_data(
            tblId='DT_1C0002', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if income_data:
            economic_data['per_capita_income'] = self.process_kosis_data(income_data, '1인당 개인소득')
        
        # 4. 소비자물가지수
        print("4. 소비자물가지수 수집 중...")
        cpi_data = self.get_kosis_data(
            tblId='DT_1J20001', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if cpi_data:
            economic_data['cpi'] = self.process_kosis_data(cpi_data, '소비자물가지수')
        
        return economic_data
    
    def collect_social_indicators(self):
        """사회지표 수집"""
        print("\n=== 사회지표 수집 ===")
        
        social_data = {}
        
        # 1. 인구증가율
        print("1. 인구증가율 수집 중...")
        population_data = self.get_kosis_data(
            tblId='DT_1B04001', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if population_data:
            social_data['population_growth'] = self.process_kosis_data(population_data, '인구증가율')
        
        # 2. 고등학교 진학률
        print("2. 고등학교 진학률 수집 중...")
        education_data = self.get_kosis_data(
            tblId='DT_1ED0001', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if education_data:
            social_data['high_school_enrollment'] = self.process_kosis_data(education_data, '고등학교 진학률')
        
        # 3. 의료기관 수
        print("3. 의료기관 수 수집 중...")
        medical_data = self.get_kosis_data(
            tblId='DT_1YL0001', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if medical_data:
            social_data['medical_facilities'] = self.process_kosis_data(medical_data, '의료기관 수')
        
        return social_data
    
    def collect_environmental_indicators(self):
        """환경지표 수집"""
        print("\n=== 환경지표 수집 ===")
        
        environmental_data = {}
        
        # 1. 대기질 지수
        print("1. 대기질 지수 수집 중...")
        air_quality_data = self.get_kosis_data(
            tblId='DT_1YL0002', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if air_quality_data:
            environmental_data['air_quality'] = self.process_kosis_data(air_quality_data, '대기질 지수')
        
        # 2. 도시공원 면적
        print("2. 도시공원 면적 수집 중...")
        park_data = self.get_kosis_data(
            tblId='DT_1YL0003', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if park_data:
            environmental_data['urban_parks'] = self.process_kosis_data(park_data, '도시공원 면적')
        
        return environmental_data
    
    def collect_infrastructure_indicators(self):
        """인프라지표 수집"""
        print("\n=== 인프라지표 수집 ===")
        
        infrastructure_data = {}
        
        # 1. 도로 포장률
        print("1. 도로 포장률 수집 중...")
        road_data = self.get_kosis_data(
            tblId='DT_1YL0004', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if road_data:
            infrastructure_data['road_pavement'] = self.process_kosis_data(road_data, '도로 포장률')
        
        # 2. 인터넷 보급률
        print("2. 인터넷 보급률 수집 중...")
        internet_data = self.get_kosis_data(
            tblId='DT_1YL0005', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if internet_data:
            infrastructure_data['internet_penetration'] = self.process_kosis_data(internet_data, '인터넷 보급률')
        
        return infrastructure_data
    
    def collect_innovation_indicators(self):
        """혁신지표 수집"""
        print("\n=== 혁신지표 수집 ===")
        
        innovation_data = {}
        
        # 1. 특허 출원 건수
        print("1. 특허 출원 건수 수집 중...")
        patent_data = self.get_kosis_data(
            tblId='DT_1YL0006', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if patent_data:
            innovation_data['patent_applications'] = self.process_kosis_data(patent_data, '특허 출원 건수')
        
        # 2. R&D 투자비
        print("2. R&D 투자비 수집 중...")
        rnd_data = self.get_kosis_data(
            tblId='DT_1YL0007', prdSe='Y', startPrdDe='1995', endPrdDe='2019',
            orgId='101', prdSe2='A00'
        )
        
        if rnd_data:
            innovation_data['rnd_investment'] = self.process_kosis_data(rnd_data, 'R&D 투자비')
        
        return innovation_data
    
    def process_kosis_data(self, data, indicator_name):
        """KOSIS 데이터 전처리"""
        try:
            # 데이터프레임으로 변환
            df = pd.DataFrame(data)
            
            # 컬럼명 정리
            if 'PRD_DE' in df.columns:
                df['year'] = df['PRD_DE'].astype(int)
            if 'C1_NM' in df.columns:
                df['region'] = df['C1_NM']
            if 'DT' in df.columns:
                df['value'] = pd.to_numeric(df['DT'], errors='coerce')
            
            # 필요한 컬럼만 선택
            if all(col in df.columns for col in ['year', 'region', 'value']):
                result_df = df[['year', 'region', 'value']].copy()
                result_df['indicator'] = indicator_name
                return result_df
            else:
                print(f"❌ {indicator_name}: 필요한 컬럼이 없습니다.")
                return None
                
        except Exception as e:
            print(f"❌ {indicator_name} 데이터 처리 실패: {e}")
            return None
    
    def create_comprehensive_dataset(self, all_indicators):
        """종합 데이터셋 생성"""
        print("\n=== 종합 데이터셋 생성 ===")
        
        # 모든 지표 데이터 통합
        combined_data = []
        
        for category, indicators in all_indicators.items():
            for indicator_name, df in indicators.items():
                if df is not None and not df.empty:
                    # 지역명 정규화
                    df['region_normalized'] = df['region'].map(self.regions)
                    
                    # 결측값 처리
                    df = df.dropna(subset=['value', 'region_normalized'])
                    
                    if not df.empty:
                        combined_data.append(df)
        
        if combined_data:
            # 데이터 통합
            comprehensive_df = pd.concat(combined_data, ignore_index=True)
            
            # 피벗 테이블 생성 (지역별, 연도별, 지표별)
            pivot_df = comprehensive_df.pivot_table(
                index=['year', 'region_normalized'],
                columns='indicator',
                values='value',
                aggfunc='mean'
            ).reset_index()
            
            # 지역명 매핑
            region_mapping = {v: k for k, v in self.regions.items()}
            pivot_df['region'] = pivot_df['region_normalized'].map(region_mapping)
            
            # 컬럼명 정리
            pivot_df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in pivot_df.columns]
            
            print(f"✅ 종합 데이터셋 생성 완료: {pivot_df.shape}")
            return pivot_df
        else:
            print("❌ 통합할 데이터가 없습니다.")
            return None
    
    def collect_all_indicators(self):
        """모든 지표 수집"""
        print("🚀 통계청 KOSIS OPEN API 데이터 수집 시작")
        
        all_indicators = {}
        
        # 1. 경제지표 수집
        all_indicators['economic'] = self.collect_economic_indicators()
        time.sleep(1)  # API 호출 간격 조절
        
        # 2. 사회지표 수집
        all_indicators['social'] = self.collect_social_indicators()
        time.sleep(1)
        
        # 3. 환경지표 수집
        all_indicators['environmental'] = self.collect_environmental_indicators()
        time.sleep(1)
        
        # 4. 인프라지표 수집
        all_indicators['infrastructure'] = self.collect_infrastructure_indicators()
        time.sleep(1)
        
        # 5. 혁신지표 수집
        all_indicators['innovation'] = self.collect_innovation_indicators()
        
        # 6. 종합 데이터셋 생성
        comprehensive_df = self.create_comprehensive_dataset(all_indicators)
        
        if comprehensive_df is not None:
            # 데이터 저장
            comprehensive_df.to_csv('kosis_comprehensive_indicators.csv', index=False, encoding='utf-8-sig')
            print("✅ KOSIS 종합 지표 데이터 저장: kosis_comprehensive_indicators.csv")
            
            # 수집 결과 요약
            self.print_collection_summary(all_indicators, comprehensive_df)
            
            return comprehensive_df
        else:
            print("❌ 데이터 수집 실패")
            return None
    
    def print_collection_summary(self, all_indicators, comprehensive_df):
        """수집 결과 요약 출력"""
        print(f"\n=== 데이터 수집 결과 요약 ===")
        
        total_indicators = 0
        successful_indicators = 0
        
        for category, indicators in all_indicators.items():
            print(f"\n📊 {category.upper()} 지표:")
            for indicator_name, df in indicators.items():
                total_indicators += 1
                if df is not None and not df.empty:
                    successful_indicators += 1
                    print(f"  ✅ {indicator_name}: {len(df)}개 데이터")
                else:
                    print(f"  ❌ {indicator_name}: 수집 실패")
        
        print(f"\n📈 종합 통계:")
        print(f"  - 총 지표 수: {total_indicators}개")
        print(f"  - 성공 수집: {successful_indicators}개")
        print(f"  - 성공률: {successful_indicators/total_indicators*100:.1f}%")
        print(f"  - 종합 데이터셋: {comprehensive_df.shape}")
        print(f"  - 지역 수: {comprehensive_df['region'].nunique()}개")
        print(f"  - 연도 범위: {comprehensive_df['year'].min()}~{comprehensive_df['year'].max()}")

def main():
    """메인 실행 함수"""
    print("=== 통계청 KOSIS OPEN API 데이터 수집기 ===")
    
    try:
        # KOSIS 데이터 수집기 초기화
        collector = KosisDataCollector()
        
        # 모든 지표 수집
        comprehensive_data = collector.collect_all_indicators()
        
        if comprehensive_data is not None:
            print(f"\n✅ KOSIS 데이터 수집 완료!")
            print(f"📊 수집된 지표: {comprehensive_data.shape[1]-3}개")
            print(f"🌍 지역 수: {comprehensive_data['region'].nunique()}개")
            print(f"📅 연도 범위: {comprehensive_data['year'].min()}~{comprehensive_data['year'].max()}")
            
            # 데이터 미리보기
            print(f"\n📋 데이터 미리보기:")
            print(comprehensive_data.head())
            
            return comprehensive_data
        else:
            print("❌ 데이터 수집에 실패했습니다.")
            return None
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None

if __name__ == "__main__":
    main()
