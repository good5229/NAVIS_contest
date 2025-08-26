#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지역간 재정 자립도 격차 분석기
재정 자립도 = 지방세 수입 / 총 재정수입 × 100
"""

import requests
import pandas as pd
import numpy as np
import sqlite3
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fiscal_autonomy_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class FiscalAutonomyAnalyzer:
    def __init__(self):
        self.db_path = 'fiscal_autonomy_data.db'
        self.regions = [
            '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시',
            '세종특별자치시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', 
            '경상북도', '경상남도', '제주특별자치도'
        ]
        self.init_database()
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 재정 자립도 데이터 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fiscal_autonomy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                year INTEGER NOT NULL,
                local_tax_revenue REAL,
                total_revenue REAL,
                fiscal_autonomy_ratio REAL,
                fiscal_power_index REAL,
                per_capita_local_tax REAL,
                transfer_dependency_ratio REAL,
                local_debt_amount REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(region, year)
            )
        ''')
        
        # 재정 격차 지수 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fiscal_gap_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                max_min_gap REAL,
                standard_deviation REAL,
                gini_coefficient REAL,
                top_bottom_ratio REAL,
                fiscal_balance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(year)
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("데이터베이스 초기화 완료")
    
    def collect_fiscal_data(self, year: int):
        """재정 데이터 수집 (시뮬레이션 데이터)"""
        logging.info(f"{year}년 재정 데이터 수집 시작")
        
        # 실제 데이터 패턴을 기반으로 한 시뮬레이션 데이터 생성
        fiscal_data = []
        
        for region in self.regions:
            # 지역별 특성에 따른 재정 자립도 차이 반영
            if region in ['서울특별시', '경기도']:
                base_autonomy = 0.85 + np.random.normal(0, 0.05)
                per_capita_tax = 800000 + np.random.normal(0, 50000)
            elif region in ['부산광역시', '대구광역시', '인천광역시']:
                base_autonomy = 0.70 + np.random.normal(0, 0.08)
                per_capita_tax = 600000 + np.random.normal(0, 40000)
            elif region in ['광주광역시', '대전광역시', '울산광역시']:
                base_autonomy = 0.65 + np.random.normal(0, 0.10)
                per_capita_tax = 550000 + np.random.normal(0, 35000)
            elif region == '세종특별자치시':
                base_autonomy = 0.75 + np.random.normal(0, 0.06)
                per_capita_tax = 700000 + np.random.normal(0, 45000)
            else:  # 도 지역
                base_autonomy = 0.45 + np.random.normal(0, 0.15)
                per_capita_tax = 400000 + np.random.normal(0, 30000)
            
            # 연도별 변화 패턴 적용
            year_factor = 1 + (year - 1997) * 0.01  # 장기적 개선 추세
            autonomy_ratio = min(0.95, max(0.20, base_autonomy * year_factor))
            
            # 인구 대비 지방세 수입 계산
            population = self.get_region_population(region, year)
            local_tax_revenue = per_capita_tax * population * autonomy_ratio
            
            # 총 재정수입 계산 (지방세 + 이전지원금 + 기타)
            total_revenue = local_tax_revenue / autonomy_ratio
            
            # 기타 지표 계산
            fiscal_power_index = autonomy_ratio * 100 + np.random.normal(0, 5)
            transfer_dependency = (1 - autonomy_ratio) * 100
            local_debt = total_revenue * (0.3 + np.random.normal(0, 0.1))
            
            fiscal_data.append({
                'region': region,
                'year': year,
                'local_tax_revenue': local_tax_revenue,
                'total_revenue': total_revenue,
                'fiscal_autonomy_ratio': autonomy_ratio,
                'fiscal_power_index': fiscal_power_index,
                'per_capita_local_tax': per_capita_tax,
                'transfer_dependency_ratio': transfer_dependency,
                'local_debt_amount': local_debt
            })
        
        return fiscal_data
    
    def get_region_population(self, region: str, year: int) -> int:
        """지역별 인구 데이터 (기존 데이터 활용)"""
        # 기존 인구 데이터 패턴 활용
        base_populations = {
            '서울특별시': 9500000,
            '부산광역시': 3300000,
            '대구광역시': 2400000,
            '인천광역시': 2900000,
            '광주광역시': 1450000,
            '대전광역시': 1500000,
            '울산광역시': 1100000,
            '세종특별자치시': 350000,
            '경기도': 13500000,
            '강원도': 1500000,
            '충청북도': 1600000,
            '충청남도': 2100000,
            '전라북도': 1800000,
            '전라남도': 1900000,
            '경상북도': 2600000,
            '경상남도': 3300000,
            '제주특별자치도': 670000
        }
        
        base_pop = base_populations.get(region, 1000000)
        # 연도별 인구 변화 패턴 적용
        year_factor = 1 + (year - 1997) * 0.005  # 완만한 증가 추세
        return int(base_pop * year_factor)
    
    def calculate_fiscal_gap_index(self, year: int, fiscal_data: List[Dict]):
        """재정 격차 지수 계산"""
        autonomy_ratios = [data['fiscal_autonomy_ratio'] for data in fiscal_data]
        
        # 최고-최저 격차
        max_min_gap = max(autonomy_ratios) - min(autonomy_ratios)
        
        # 표준편차
        standard_deviation = np.std(autonomy_ratios)
        
        # 지니계수 (불평등도)
        sorted_ratios = sorted(autonomy_ratios)
        n = len(sorted_ratios)
        cumsum = np.cumsum(sorted_ratios)
        gini_coefficient = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # 상위/하위 평균 비율
        top_third = np.mean(sorted_ratios[-6:])  # 상위 1/3
        bottom_third = np.mean(sorted_ratios[:6])  # 하위 1/3
        top_bottom_ratio = top_third / bottom_third if bottom_third > 0 else 0
        
        # 재정 균형 점수 (높을수록 균형)
        fiscal_balance_score = max(0, 100 - max_min_gap * 200 - standard_deviation * 100)
        
        return {
            'year': year,
            'max_min_gap': max_min_gap,
            'standard_deviation': standard_deviation,
            'gini_coefficient': gini_coefficient,
            'top_bottom_ratio': top_bottom_ratio,
            'fiscal_balance_score': fiscal_balance_score
        }
    
    def save_to_db(self, fiscal_data: List[Dict], gap_index: Dict):
        """데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        
        # 재정 자립도 데이터 저장
        fiscal_df = pd.DataFrame(fiscal_data)
        fiscal_df.to_sql('fiscal_autonomy', conn, if_exists='append', index=False)
        
        # 재정 격차 지수 저장
        gap_df = pd.DataFrame([gap_index])
        gap_df.to_sql('fiscal_gap_index', conn, if_exists='append', index=False)
        
        conn.close()
        logging.info(f"데이터베이스 저장 완료")
    
    def collect_all_data(self, start_year: int = 1997, end_year: int = 2025):
        """전체 데이터 수집"""
        logging.info(f"재정 자립도 데이터 수집 시작: {start_year}년 ~ {end_year}년")
        
        for year in range(start_year, end_year + 1):
            logging.info(f"=== {year}년 재정 데이터 수집 ===")
            
            # 재정 데이터 수집
            fiscal_data = self.collect_fiscal_data(year)
            
            # 재정 격차 지수 계산
            gap_index = self.calculate_fiscal_gap_index(year, fiscal_data)
            
            # 데이터베이스 저장
            self.save_to_db(fiscal_data, gap_index)
            
            time.sleep(0.5)  # 처리 간격
        
        logging.info("전체 재정 자립도 데이터 수집 완료")
    
    def create_visualizations(self):
        """시각화 생성"""
        conn = sqlite3.connect(self.db_path)
        
        # 데이터 로드
        fiscal_df = pd.read_sql_query("SELECT * FROM fiscal_autonomy", conn)
        gap_df = pd.read_sql_query("SELECT * FROM fiscal_gap_index", conn)
        
        conn.close()
        
        # 1. 재정 자립도 트렌드 차트
        self.create_fiscal_autonomy_trend(fiscal_df)
        
        # 2. 재정 격차 지수 차트
        self.create_fiscal_gap_trend(gap_df)
        
        # 3. 지역별 재정 자립도 비교 차트
        self.create_regional_comparison(fiscal_df)
        
        # 4. 상관관계 분석 차트
        self.create_correlation_analysis(fiscal_df)
    
    def create_fiscal_autonomy_trend(self, df: pd.DataFrame):
        """재정 자립도 트렌드 차트"""
        fig = go.Figure()
        
        for region in self.regions:
            region_data = df[df['region'] == region]
            fig.add_trace(go.Scatter(
                x=region_data['year'],
                y=region_data['fiscal_autonomy_ratio'] * 100,
                mode='lines+markers',
                name=region,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='지역별 재정 자립도 변화 추이 (1997-2025)',
            xaxis_title='연도',
            yaxis_title='재정 자립도 (%)',
            hovermode='x unified',
            height=600
        )
        
        fig.write_html('fiscal_autonomy_trend.html')
        logging.info("재정 자립도 트렌드 차트 생성 완료")
    
    def create_fiscal_gap_trend(self, df: pd.DataFrame):
        """재정 격차 지수 트렌드 차트"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('재정 균형 점수', '최고-최저 격차', '표준편차', '지니계수'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 재정 균형 점수
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['fiscal_balance_score'], 
                      mode='lines+markers', name='재정 균형 점수',
                      line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        # 최고-최저 격차
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['max_min_gap'] * 100, 
                      mode='lines+markers', name='최고-최저 격차 (%)',
                      line=dict(color='red', width=3)),
            row=1, col=2
        )
        
        # 표준편차
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['standard_deviation'] * 100, 
                      mode='lines+markers', name='표준편차 (%)',
                      line=dict(color='blue', width=3)),
            row=2, col=1
        )
        
        # 지니계수
        fig.add_trace(
            go.Scatter(x=df['year'], y=df['gini_coefficient'], 
                      mode='lines+markers', name='지니계수',
                      line=dict(color='purple', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="재정 격차 지수 변화 추이")
        fig.write_html('fiscal_gap_trend.html')
        logging.info("재정 격차 지수 차트 생성 완료")
    
    def create_regional_comparison(self, df: pd.DataFrame):
        """지역별 재정 자립도 비교 차트"""
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].sort_values('fiscal_autonomy_ratio', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=latest_data['fiscal_autonomy_ratio'] * 100,
                y=latest_data['region'],
                orientation='h',
                marker_color='lightblue',
                text=[f"{val:.1f}%" for val in latest_data['fiscal_autonomy_ratio'] * 100],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'{latest_year}년 지역별 재정 자립도 비교',
            xaxis_title='재정 자립도 (%)',
            yaxis_title='지역',
            height=600
        )
        
        fig.write_html('fiscal_regional_comparison.html')
        logging.info("지역별 재정 자립도 비교 차트 생성 완료")
    
    def create_correlation_analysis(self, df: pd.DataFrame):
        """상관관계 분석 차트"""
        # 재정 자립도와 1인당 지방세 상관관계
        fig = px.scatter(
            df, x='fiscal_autonomy_ratio', y='per_capita_local_tax',
            color='region', size='total_revenue',
            title='재정 자립도와 1인당 지방세 상관관계',
            labels={'fiscal_autonomy_ratio': '재정 자립도', 'per_capita_local_tax': '1인당 지방세 (원)'}
        )
        
        fig.write_html('fiscal_correlation_analysis.html')
        logging.info("상관관계 분석 차트 생성 완료")
    
    def export_results(self):
        """결과 데이터 내보내기"""
        conn = sqlite3.connect(self.db_path)
        
        fiscal_df = pd.read_sql_query("SELECT * FROM fiscal_autonomy", conn)
        gap_df = pd.read_sql_query("SELECT * FROM fiscal_gap_index", conn)
        
        conn.close()
        
        # CSV 파일로 저장
        fiscal_df.to_csv('fiscal_autonomy_data.csv', index=False, encoding='utf-8-sig')
        gap_df.to_csv('fiscal_gap_index_data.csv', index=False, encoding='utf-8-sig')
        
        logging.info("결과 데이터 내보내기 완료")

if __name__ == "__main__":
    analyzer = FiscalAutonomyAnalyzer()
    analyzer.collect_all_data(1997, 2025)
    analyzer.create_visualizations()
    analyzer.export_results()
