#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통계청 KOSIS API를 사용한 재정자립도 데이터 수집기
데이터 출처: https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1YL20921&conn_path=I2
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kosis_fiscal_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class KosisFiscalDataCollector:
    def __init__(self):
        # KOSIS OpenAPI 설정
        import os
        self.api_key = os.environ.get("KOSIS_API_KEY", "")
        self.base_url = "https://kosis.kr/openapi/statApi.do"
        
        # 재정자립도 통계 테이블 정보 (수정된 버전)
        self.org_id = "101"  # 통계청
        # 실제 KOSIS 테이블 ID 확인 필요 - 임시로 다른 ID 시도
        self.tbl_id_autonomy = "DT_1YL20921"  # 지방자치단체별 재정자립도
        self.tbl_id_independence = "DT_1YL20891"  # 지방자치단체별 재정자주도
        
        # KOSIS API 올바른 엔드포인트
        self.base_url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
        
        # 지역 매핑 (KOSIS 코드 -> 우리 프로젝트 지역명)
        self.region_mapping = {
            "00": "전국",
            "11": "서울특별시",
            "21": "부산광역시", 
            "22": "대구광역시",
            "23": "인천광역시",
            "24": "광주광역시",
            "25": "대전광역시",
            "26": "울산광역시",
            "29": "세종특별자치시",
            "31": "경기도",
            "32": "강원도",
            "33": "충청북도",
            "34": "충청남도",
            "35": "전라북도",
            "36": "전라남도",
            "37": "경상북도",
            "38": "경상남도",
            "39": "제주특별자치도"
        }
        
    def get_kosis_data(self, start_year: int = 2001, end_year: int = 2025) -> pd.DataFrame:
        """KOSIS API에서 재정자립도 및 재정자주도 데이터 수집"""
        logging.info(f"KOSIS API에서 재정 데이터 수집 시작: {start_year}년 ~ {end_year}년")
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            logging.info(f"{year}년 데이터 수집 중...")
            
            # 재정자립도 데이터 수집
            autonomy_data = self._get_single_year_data(year, self.tbl_id_autonomy, "재정자립도")
            independence_data = self._get_single_year_data(year, self.tbl_id_independence, "재정자주도")
            
            # 데이터 병합
            if autonomy_data and independence_data:
                merged_data = self._merge_autonomy_independence_data(autonomy_data, independence_data, year)
                all_data.extend(merged_data)
            
            # API 호출 제한 방지
            time.sleep(2)
        
        df = pd.DataFrame(all_data)
        logging.info(f"전체 데이터 수집 완료: {len(df)}개 레코드")
        
        return df
    
    def _get_single_year_data(self, year: int, tbl_id: str, data_type: str) -> List[Dict]:
        """단일 연도 데이터 수집"""
        try:
            # KOSIS API 요청 파라미터 (올바른 버전)
            params = {
                "apiKey": self.api_key,
                "method": "getList",
                "format": "json",
                "jsonVD": "Y",
                "prdSe": "Y",  # 연도별
                "startPrdDe": str(year),
                "endPrdDe": str(year),
                "orgId": self.org_id,
                "tblId": tbl_id,
                "itmId": "T10 T20",  # 항목 ID
                "objL1": "00 11 21 22 23 24 25 26 29 31 32 33 34 35 36 37 38 39",  # 지역 코드
                "objL2": "",
                "objL3": "",
                "objL4": "",
                "objL5": "",
                "objL6": "",
                "objL7": "",
                "objL8": "",
                "outputFields": "TBL_ID OBJ_ID PRD_SE LST_CHN_DE"
            }
            
            logging.info(f"API 요청 URL: {self.base_url}")
            logging.info(f"API 요청 파라미터: {params}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            
            logging.info(f"응답 상태 코드: {response.status_code}")
            logging.info(f"응답 헤더: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logging.info(f"응답 데이터 구조: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # API 응답이 배열 형태로 직접 반환됨
                    if isinstance(data, list):
                        year_data = []
                        for item in data:
                            region_code = item.get('C1', '')
                            region_name = item.get('C1_NM', '')
                            value = float(item.get('DT', 0)) if item.get('DT') else 0
                            
                            # 전국 데이터는 제외
                            if region_name == '전국':
                                continue
                            
                            # 지역명이 비어있으면 지역 코드로 매핑
                            if not region_name and region_code:
                                region_name = self.region_mapping.get(region_code, f'지역코드_{region_code}')
                            
                            year_data.append({
                                'year': year,
                                'region_code': region_code,
                                'region': region_name,
                                'fiscal_autonomy_ratio': value / 100 if data_type == "재정자립도" else 0,
                                'fiscal_independence_ratio': value / 100 if data_type == "재정자주도" else 0,
                                'data_type': data_type,
                                'source': 'KOSIS'
                            })
                        
                        logging.info(f"{year}년 {data_type} 데이터 수집 완료: {len(year_data)}개 지역")
                        return year_data
                    else:
                        logging.warning(f"{year}년 {data_type} 데이터 구조가 예상과 다름")
                        return []
                        
                except json.JSONDecodeError as e:
                    logging.error(f"{year}년 {data_type} JSON 파싱 오류: {str(e)}")
                    logging.error(f"응답 내용: {response.text[:500]}")
                    return []
            else:
                logging.error(f"{year}년 {data_type} 데이터 수집 실패: {response.status_code}")
                logging.error(f"응답 내용: {response.text[:500]}")
                return []
                
        except Exception as e:
            logging.error(f"{year}년 {data_type} 데이터 수집 중 오류: {str(e)}")
            return []
    
    def _merge_autonomy_independence_data(self, autonomy_data: List[Dict], independence_data: List[Dict], year: int) -> List[Dict]:
        """재정자립도와 재정자주도 데이터 병합"""
        merged_data = []
        
        # 재정자립도 데이터를 딕셔너리로 변환
        autonomy_dict = {item['region']: item['fiscal_autonomy_ratio'] for item in autonomy_data}
        independence_dict = {item['region']: item['fiscal_independence_ratio'] for item in independence_data}
        
        # 모든 지역에 대해 데이터 병합
        all_regions = set(autonomy_dict.keys()) | set(independence_dict.keys())
        
        for region in all_regions:
            merged_data.append({
                'year': year,
                'region': region,
                'fiscal_autonomy_ratio': autonomy_dict.get(region, 0),
                'fiscal_independence_ratio': independence_dict.get(region, 0),
                'source': 'KOSIS'
            })
        
        return merged_data
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "data/fiscal_autonomy/kosis_fiscal_autonomy_data.csv"):
        """데이터를 CSV 파일로 저장"""
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logging.info(f"데이터 저장 완료: {filename}")
    
    def save_to_json(self, df: pd.DataFrame, filename: str = "kosis_fiscal_autonomy_data.json"):
        """데이터를 JSON 파일로 저장 (대시보드용)"""
        # 연도별로 데이터 구조화
        data_by_year = {}
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            data_by_year[str(year)] = []  # int64를 str로 변환
            
            for _, row in year_data.iterrows():
                data_by_year[str(year)].append({
                    'region': row['region'],
                    'fiscal_autonomy_ratio': float(row['fiscal_autonomy_ratio']),  # numpy 타입을 float로 변환
                    'fiscal_independence_ratio': float(row['fiscal_independence_ratio']),  # numpy 타입을 float로 변환
                    'per_capita_local_tax': 0,  # KOSIS에서 제공하지 않음
                    'total_revenue': 0  # KOSIS에서 제공하지 않음
                })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_by_year, f, ensure_ascii=False, indent=2)
        
        logging.info(f"JSON 데이터 저장 완료: {filename}")
    
    def create_sample_data(self) -> Dict:
        """샘플 데이터 생성을 허용하지 않습니다."""
        raise RuntimeError("샘플 데이터 생성이 금지되었습니다. 실제 KOSIS API를 사용하세요.")

if __name__ == "__main__":
    collector = KosisFiscalDataCollector()
    
    try:
        # 실제 KOSIS API 데이터 수집 시도
        df = collector.get_kosis_data(2001, 2025)
        collector.save_to_csv(df)
        collector.save_to_json(df)
        
    except Exception as e:
        logging.error(f"KOSIS API 접근 실패: {str(e)}")
        raise
