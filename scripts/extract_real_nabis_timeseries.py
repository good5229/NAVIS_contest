#!/usr/bin/env python3
"""
실제 NABIS 시계열 데이터 추출 스크립트
- NABIS Excel 파일에서 2016-2019년 지역별 데이터 추출
- 그레인저 인과관계 검정을 위한 데이터 준비
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_navis_timeseries():
    """NABIS Excel 파일에서 실제 시계열 데이터 추출"""
    
    navis_file = Path("data/navis/1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx")
    
    if not navis_file.exists():
        logger.error(f"NABIS 파일을 찾을 수 없습니다: {navis_file}")
        return None
    
    try:
        # Excel 파일의 모든 시트 확인
        excel_file = pd.ExcelFile(navis_file)
        logger.info(f"사용 가능한 시트: {excel_file.sheet_names}")
        
        # 각 시트를 확인하여 시계열 데이터 찾기
        navis_data = {}
        
        for sheet_name in excel_file.sheet_names:
            logger.info(f"시트 '{sheet_name}' 분석 중...")
            
            try:
                df = pd.read_excel(navis_file, sheet_name=sheet_name)
                logger.info(f"시트 크기: {df.shape}")
                logger.info(f"컬럼: {df.columns.tolist()}")
                
                # 첫 몇 행 확인
                print(f"\n=== 시트: {sheet_name} ===")
                print(df.head())
                
                # 연도 컬럼이 있는지 확인
                year_columns = [col for col in df.columns if str(col).isdigit() and 2016 <= int(str(col)) <= 2019]
                if year_columns:
                    logger.info(f"발견된 연도 컬럼: {year_columns}")
                
            except Exception as e:
                logger.warning(f"시트 '{sheet_name}' 읽기 실패: {e}")
                continue
        
        return navis_data
        
    except Exception as e:
        logger.error(f"NABIS 데이터 추출 실패: {e}")
        return None

def main():
    """메인 실행 함수"""
    logger.info("실제 NABIS 시계열 데이터 추출 시작")
    
    navis_data = extract_navis_timeseries()
    
    if navis_data is not None:
        logger.info("NABIS 데이터 추출 완료")
    else:
        logger.error("NABIS 데이터 추출 실패")

if __name__ == "__main__":
    main()
