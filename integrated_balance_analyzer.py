#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새 정부 국가 균형성장 비전을 위한 종합 균형발전 분석 시스템
통합 균형발전 지수 분석 및 정책 제언 생성
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_balance_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class IntegratedBalanceAnalyzer:
    def __init__(self):
        self.db_path = 'integrated_balance_data.db'
        self.regions = [
            '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시',
            '세종특별자치시', '경기도', '강원도', '충청북도', '충청남도', '전라북도', '전라남도', 
            '경상북도', '경상남도', '제주특별자치도'
        ]
        self.load_data()
        
    def load_data(self):
        """데이터베이스에서 데이터 로드"""
        conn = sqlite3.connect(self.db_path)
        
        self.economic_df = pd.read_sql_query("SELECT * FROM economic_balance", conn)
        self.quality_df = pd.read_sql_query("SELECT * FROM quality_of_life_balance", conn)
        self.environmental_df = pd.read_sql_query("SELECT * FROM environmental_balance", conn)
        self.welfare_df = pd.read_sql_query("SELECT * FROM welfare_balance", conn)
        self.integrated_df = pd.read_sql_query("SELECT * FROM integrated_balance_index", conn)
        
        conn.close()
        logging.info("데이터 로드 완료")
    
    def analyze_regional_imbalance(self, year: int = 2025) -> Dict:
        """지역 간 불균형 분석"""
        logging.info(f"지역 간 불균형 분석 시작: {year}년")
        
        year_data = self.integrated_df[self.integrated_df['year'] == year]
        
        # 통합 점수 분석
        scores = year_data['integrated_score'].values
        max_score = np.max(scores)
        min_score = np.min(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # 최고/최저 지역
        best_region = year_data.loc[year_data['integrated_score'].idxmax(), 'region']
        worst_region = year_data.loc[year_data['integrated_score'].idxmin(), 'region']
        
        # 격차 분석
        score_gap = max_score - min_score
        coefficient_variation = std_score / mean_score
        
        # 상위/하위 30% 분석
        sorted_scores = np.sort(scores)
        top_30_count = int(len(scores) * 0.3)
        bottom_30_count = int(len(scores) * 0.3)
        
        top_30_avg = np.mean(sorted_scores[-top_30_count:])
        bottom_30_avg = np.mean(sorted_scores[:bottom_30_count])
        top_bottom_gap = top_30_avg - bottom_30_avg
        
        # 균형 수준 분포
        balance_distribution = year_data['balance_level'].value_counts().to_dict()
        
        analysis_result = {
            'year': year,
            'total_regions': len(scores),
            'max_score': round(max_score, 2),
            'min_score': round(min_score, 2),
            'mean_score': round(mean_score, 2),
            'std_score': round(std_score, 2),
            'score_gap': round(score_gap, 2),
            'coefficient_variation': round(coefficient_variation, 3),
            'top_30_avg': round(top_30_avg, 2),
            'bottom_30_avg': round(bottom_30_avg, 2),
            'top_bottom_gap': round(top_bottom_gap, 2),
            'best_region': best_region,
            'worst_region': worst_region,
            'balance_distribution': balance_distribution
        }
        
        logging.info(f"지역 간 불균형 분석 완료: 격차 {score_gap:.2f}")
        return analysis_result
    
    def analyze_domain_imbalance(self, year: int = 2025) -> Dict:
        """영역별 불균형 분석"""
        logging.info(f"영역별 불균형 분석 시작: {year}년")
        
        year_data = self.integrated_df[self.integrated_df['year'] == year]
        
        domains = ['economic_score', 'quality_of_life_score', 'environmental_score', 'welfare_score']
        domain_names = ['경제', '삶의질', '환경', '복지']
        
        domain_analysis = {}
        
        for domain, domain_name in zip(domains, domain_names):
            scores = year_data[domain].values
            
            domain_analysis[domain_name] = {
                'mean': round(np.mean(scores), 2),
                'std': round(np.std(scores), 2),
                'max': round(np.max(scores), 2),
                'min': round(np.min(scores), 2),
                'gap': round(np.max(scores) - np.min(scores), 2),
                'cv': round(np.std(scores) / np.mean(scores), 3),
                'best_region': year_data.loc[year_data[domain].idxmax(), 'region'],
                'worst_region': year_data.loc[year_data[domain].idxmin(), 'region']
            }
        
        logging.info("영역별 불균형 분석 완료")
        return domain_analysis
    
    def analyze_temporal_trends(self, start_year: int = 2015, end_year: int = 2025) -> Dict:
        """시계열 추이 분석"""
        logging.info(f"시계열 추이 분석 시작: {start_year}년 ~ {end_year}년")
        
        trend_data = self.integrated_df[
            (self.integrated_df['year'] >= start_year) & 
            (self.integrated_df['year'] <= end_year)
        ]
        
        # 연도별 평균 점수 추이
        yearly_means = trend_data.groupby('year')['integrated_score'].mean()
        
        # 연도별 격차 추이
        yearly_gaps = trend_data.groupby('year')['integrated_score'].agg(['max', 'min'])
        yearly_gaps['gap'] = yearly_gaps['max'] - yearly_gaps['min']
        
        # 연도별 균형 수준 분포 변화
        balance_trends = trend_data.groupby(['year', 'balance_level']).size().unstack(fill_value=0)
        
        trend_analysis = {
            'years': list(yearly_means.index),
            'mean_scores': [round(score, 2) for score in yearly_means.values],
            'gaps': [round(gap, 2) for gap in yearly_gaps['gap'].values],
            'balance_trends': balance_trends.to_dict(),
            'improvement_rate': round(
                (yearly_means.iloc[-1] - yearly_means.iloc[0]) / yearly_means.iloc[0] * 100, 2
            )
        }
        
        logging.info(f"시계열 추이 분석 완료: 개선률 {trend_analysis['improvement_rate']}%")
        return trend_analysis
    
    def identify_policy_priorities(self, year: int = 2025) -> Dict:
        """정책 우선순위 도출"""
        logging.info(f"정책 우선순위 도출 시작: {year}년")
        
        year_data = self.integrated_df[self.integrated_df['year'] == year]
        
        # 지역별 영역별 점수 분석
        domains = ['economic_score', 'quality_of_life_score', 'environmental_score', 'welfare_score']
        domain_names = ['경제', '삶의질', '환경', '복지']
        
        priorities = {}
        
        for region in self.regions:
            region_data = year_data[year_data['region'] == region].iloc[0]
            
            # 각 영역별 점수
            domain_scores = {}
            for domain, domain_name in zip(domains, domain_names):
                domain_scores[domain_name] = region_data[domain]
            
            # 최저 점수 영역 찾기
            min_domain = min(domain_scores, key=domain_scores.get)
            min_score = domain_scores[min_domain]
            
            # 종합 점수
            integrated_score = region_data['integrated_score']
            balance_level = region_data['balance_level']
            
            priorities[region] = {
                'integrated_score': round(integrated_score, 2),
                'balance_level': balance_level,
                'weakest_domain': min_domain,
                'weakest_score': round(min_score, 2),
                'domain_scores': {k: round(v, 2) for k, v in domain_scores.items()},
                'priority_level': self.determine_priority_level(integrated_score, min_score)
            }
        
        # 우선순위별 지역 분류
        priority_regions = {
            '매우 높음': [r for r, p in priorities.items() if p['priority_level'] == '매우 높음'],
            '높음': [r for r, p in priorities.items() if p['priority_level'] == '높음'],
            '보통': [r for r, p in priorities.items() if p['priority_level'] == '보통'],
            '낮음': [r for r, p in priorities.items() if p['priority_level'] == '낮음']
        }
        
        logging.info("정책 우선순위 도출 완료")
        return {
            'region_priorities': priorities,
            'priority_regions': priority_regions
        }
    
    def determine_priority_level(self, integrated_score: float, weakest_score: float) -> str:
        """우선순위 수준 결정"""
        if integrated_score < 65 or weakest_score < 50:
            return "매우 높음"
        elif integrated_score < 75 or weakest_score < 60:
            return "높음"
        elif integrated_score < 85 or weakest_score < 70:
            return "보통"
        else:
            return "낮음"
    
    def generate_policy_recommendations(self, year: int = 2025) -> Dict:
        """정책 제언 생성"""
        logging.info(f"정책 제언 생성 시작: {year}년")
        
        # 분석 결과 수집
        regional_analysis = self.analyze_regional_imbalance(year)
        domain_analysis = self.analyze_domain_imbalance(year)
        priority_analysis = self.identify_policy_priorities(year)
        
        recommendations = {
            'overall_assessment': self.generate_overall_assessment(regional_analysis),
            'regional_policies': self.generate_regional_policies(priority_analysis),
            'domain_policies': self.generate_domain_policies(domain_analysis),
            'investment_strategy': self.generate_investment_strategy(priority_analysis),
            'implementation_roadmap': self.generate_implementation_roadmap()
        }
        
        logging.info("정책 제언 생성 완료")
        return recommendations
    
    def generate_overall_assessment(self, regional_analysis: Dict) -> Dict:
        """전체 평가 생성"""
        score_gap = regional_analysis['score_gap']
        coefficient_variation = regional_analysis['coefficient_variation']
        top_bottom_gap = regional_analysis['top_bottom_gap']
        
        if score_gap > 20 or coefficient_variation > 0.3:
            assessment_level = "매우 불균형"
            urgency = "매우 높음"
        elif score_gap > 15 or coefficient_variation > 0.25:
            assessment_level = "불균형"
            urgency = "높음"
        elif score_gap > 10 or coefficient_variation > 0.2:
            assessment_level = "보통"
            urgency = "보통"
        else:
            assessment_level = "균형"
            urgency = "낮음"
        
        return {
            'assessment_level': assessment_level,
            'urgency': urgency,
            'key_issues': [
                f"최고-최저 지역 격차: {score_gap}점",
                f"상위-하위 30% 격차: {top_bottom_gap}점",
                f"변동계수: {coefficient_variation}",
                f"최고 지역: {regional_analysis['best_region']}",
                f"최저 지역: {regional_analysis['worst_region']}"
            ]
        }
    
    def generate_regional_policies(self, priority_analysis: Dict) -> Dict:
        """지역별 정책 생성"""
        regional_policies = {}
        
        for region, priority in priority_analysis['region_priorities'].items():
            weakest_domain = priority['weakest_domain']
            priority_level = priority['priority_level']
            
            # 영역별 정책 매핑
            domain_policies = {
                '경제': [
                    '지역 특화 산업 육성',
                    '스타트업 생태계 구축',
                    '일자리 창출 프로그램',
                    '지역 금융 지원 확대'
                ],
                '삶의질': [
                    '주거환경 개선',
                    '교육 인프라 확충',
                    '의료 접근성 향상',
                    '문화시설 확대',
                    '대중교통 개선'
                ],
                '환경': [
                    '친환경 에너지 확대',
                    '대기질 개선 프로그램',
                    '녹지 확대',
                    '기후변화 적응 인프라',
                    '폐기물 재활용 확대'
                ],
                '복지': [
                    '복지시설 확충',
                    '소득지원 프로그램',
                    '건강관리 서비스',
                    '사회적 배제 해소',
                    '취약계층 지원'
                ]
            }
            
            # 우선순위별 투자 강도
            investment_intensity = {
                '매우 높음': '대폭 확대',
                '높음': '확대',
                '보통': '점진적 확대',
                '낮음': '유지'
            }
            
            regional_policies[region] = {
                'priority_level': priority_level,
                'weakest_domain': weakest_domain,
                'recommended_policies': domain_policies[weakest_domain],
                'investment_intensity': investment_intensity[priority_level],
                'expected_impact': f"{weakest_domain} 영역 {priority_level} 수준 개선 예상"
            }
        
        return regional_policies
    
    def generate_domain_policies(self, domain_analysis: Dict) -> Dict:
        """영역별 정책 생성"""
        domain_policies = {}
        
        for domain, analysis in domain_analysis.items():
            gap = analysis['gap']
            cv = analysis['cv']
            
            if gap > 15 or cv > 0.25:
                policy_focus = "격차 해소"
                intensity = "강화"
            elif gap > 10 or cv > 0.2:
                policy_focus = "균형 발전"
                intensity = "확대"
            else:
                policy_focus = "유지 관리"
                intensity = "유지"
            
            domain_policies[domain] = {
                'policy_focus': policy_focus,
                'intensity': intensity,
                'gap': gap,
                'cv': cv,
                'best_region': analysis['best_region'],
                'worst_region': analysis['worst_region'],
                'recommendations': self.get_domain_recommendations(domain, gap, cv)
            }
        
        return domain_policies
    
    def get_domain_recommendations(self, domain: str, gap: float, cv: float) -> List[str]:
        """영역별 구체적 정책 제언"""
        recommendations = {
            '경제': [
                '지역별 특화 산업 클러스터 구축',
                '스타트업 생태계 조성',
                '지역 금융 지원 확대',
                '일자리 매칭 프로그램 강화'
            ],
            '삶의질': [
                '주거환경 개선 프로그램',
                '교육 인프라 확충',
                '의료 접근성 향상',
                '문화시설 확대',
                '대중교통 개선'
            ],
            '환경': [
                '친환경 에너지 확대',
                '대기질 개선 프로그램',
                '녹지 확대',
                '기후변화 적응 인프라',
                '폐기물 재활용 확대'
            ],
            '복지': [
                '복지시설 확충',
                '소득지원 프로그램',
                '건강관리 서비스',
                '사회적 배제 해소',
                '취약계층 지원'
            ]
        }
        
        return recommendations.get(domain, [])
    
    def generate_investment_strategy(self, priority_analysis: Dict) -> Dict:
        """투자 전략 생성"""
        priority_regions = priority_analysis['priority_regions']
        
        # 우선순위별 투자 비율
        investment_ratios = {
            '매우 높음': 0.4,  # 40%
            '높음': 0.35,      # 35%
            '보통': 0.2,       # 20%
            '낮음': 0.05       # 5%
        }
        
        strategy = {
            'investment_allocation': investment_ratios,
            'priority_regions': priority_regions,
            'total_regions_by_priority': {k: len(v) for k, v in priority_regions.items()},
            'recommended_approach': '우선순위 기반 차등 투자',
            'expected_outcome': '지역 간 격차 단계적 해소'
        }
        
        return strategy
    
    def generate_implementation_roadmap(self) -> Dict:
        """실행 로드맵 생성"""
        roadmap = {
            'phase_1': {
                'period': '1-2년',
                'focus': '긴급 불균형 해소',
                'target_regions': '매우 높음 우선순위 지역',
                'key_actions': [
                    '긴급 투자 프로그램 실행',
                    '기초 인프라 확충',
                    '취약 영역 집중 지원'
                ]
            },
            'phase_2': {
                'period': '3-5년',
                'focus': '균형 발전 기반 구축',
                'target_regions': '높음 우선순위 지역',
                'key_actions': [
                    '지속가능한 발전 모델 구축',
                    '지역 특화 프로그램 확대',
                    '협력 네트워크 강화'
                ]
            },
            'phase_3': {
                'period': '6-10년',
                'focus': '포용적 성장 달성',
                'target_regions': '전체 지역',
                'key_actions': [
                    '균형발전 체계 완성',
                    '지역 간 협력 강화',
                    '지속가능한 성장 모델 확산'
                ]
            }
        }
        
        return roadmap
    
    def create_visualizations(self, year: int = 2025):
        """시각화 생성"""
        logging.info(f"시각화 생성 시작: {year}년")
        
        # 1. 지역별 통합 점수 막대 그래프
        year_data = self.integrated_df[self.integrated_df['year'] == year]
        
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 지역별 통합 점수
        plt.subplot(2, 2, 1)
        bars = plt.bar(year_data['region'], year_data['integrated_score'])
        plt.title(f'{year}년 지역별 통합 균형발전 지수', fontsize=14, fontweight='bold')
        plt.xlabel('지역')
        plt.ylabel('통합 점수')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # 색상 구분 (균형 수준별)
        colors = {'매우 균형': 'green', '균형': 'blue', '보통': 'orange', '불균형': 'red', '매우 불균형': 'darkred'}
        for bar, level in zip(bars, year_data['balance_level']):
            bar.set_color(colors.get(level, 'gray'))
        
        # 서브플롯 2: 영역별 점수 비교
        plt.subplot(2, 2, 2)
        domains = ['economic_score', 'quality_of_life_score', 'environmental_score', 'welfare_score']
        domain_names = ['경제', '삶의질', '환경', '복지']
        
        x = np.arange(len(self.regions))
        width = 0.2
        
        for i, (domain, domain_name) in enumerate(zip(domains, domain_names)):
            plt.bar(x + i*width, year_data[domain], width, label=domain_name)
        
        plt.title(f'{year}년 영역별 점수 비교', fontsize=14, fontweight='bold')
        plt.xlabel('지역')
        plt.ylabel('점수')
        plt.xticks(x + width*1.5, year_data['region'], rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 100)
        
        # 서브플롯 3: 균형 수준 분포
        plt.subplot(2, 2, 3)
        balance_counts = year_data['balance_level'].value_counts()
        plt.pie(balance_counts.values, labels=balance_counts.index, autopct='%1.1f%%')
        plt.title(f'{year}년 균형 수준 분포', fontsize=14, fontweight='bold')
        
        # 서브플롯 4: 시계열 추이 (2015-2025)
        plt.subplot(2, 2, 4)
        trend_data = self.integrated_df.groupby('year')['integrated_score'].mean()
        plt.plot(trend_data.index, trend_data.values, marker='o', linewidth=2, markersize=6)
        plt.title('연도별 평균 통합 점수 추이', fontsize=14, fontweight='bold')
        plt.xlabel('연도')
        plt.ylabel('평균 점수')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'integrated_balance_analysis_{year}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"시각화 생성 완료: integrated_balance_analysis_{year}.png")
    
    def generate_report(self, year: int = 2025) -> str:
        """종합 보고서 생성"""
        logging.info(f"종합 보고서 생성 시작: {year}년")
        
        # 분석 실행
        regional_analysis = self.analyze_regional_imbalance(year)
        domain_analysis = self.analyze_domain_imbalance(year)
        priority_analysis = self.identify_policy_priorities(year)
        recommendations = self.generate_policy_recommendations(year)
        
        # 시각화 생성
        self.create_visualizations(year)
        
        # 보고서 템플릿
        report = f"""
# 새 정부 국가 균형성장 비전 종합 분석 보고서

## 📊 분석 개요
- **분석 연도**: {year}년
- **분석 지역**: {len(self.regions)}개 행정구역
- **분석 영역**: 경제, 삶의질, 환경, 복지
- **분석 방법**: 통합 균형발전 지수 기반 종합 분석

## 🎯 전체 평가
- **평가 수준**: {recommendations['overall_assessment']['assessment_level']}
- **긴급도**: {recommendations['overall_assessment']['urgency']}

### 주요 이슈
"""
        
        for issue in recommendations['overall_assessment']['key_issues']:
            report += f"- {issue}\n"
        
        report += f"""
## 📈 영역별 분석

### 경제 균형
- **평균 점수**: {domain_analysis['경제']['mean']}점
- **지역 간 격차**: {domain_analysis['경제']['gap']}점
- **최고 지역**: {domain_analysis['경제']['best_region']}
- **최저 지역**: {domain_analysis['경제']['worst_region']}

### 삶의 질 균형
- **평균 점수**: {domain_analysis['삶의질']['mean']}점
- **지역 간 격차**: {domain_analysis['삶의질']['gap']}점
- **최고 지역**: {domain_analysis['삶의질']['best_region']}
- **최저 지역**: {domain_analysis['삶의질']['worst_region']}

### 환경 균형
- **평균 점수**: {domain_analysis['환경']['mean']}점
- **지역 간 격차**: {domain_analysis['환경']['gap']}점
- **최고 지역**: {domain_analysis['환경']['best_region']}
- **최저 지역**: {domain_analysis['환경']['worst_region']}

### 복지 균형
- **평균 점수**: {domain_analysis['복지']['mean']}점
- **지역 간 격차**: {domain_analysis['복지']['gap']}점
- **최고 지역**: {domain_analysis['복지']['best_region']}
- **최저 지역**: {domain_analysis['복지']['worst_region']}

## 🎯 정책 우선순위

### 매우 높음 우선순위 지역 ({len(priority_analysis['priority_regions']['매우 높음'])}개)
"""
        
        for region in priority_analysis['priority_regions']['매우 높음']:
            priority = priority_analysis['region_priorities'][region]
            report += f"- **{region}**: {priority['weakest_domain']} 영역 집중 지원 필요 (점수: {priority['weakest_score']}점)\n"
        
        report += f"""
### 높음 우선순위 지역 ({len(priority_analysis['priority_regions']['높음'])}개)
"""
        
        for region in priority_analysis['priority_regions']['높음']:
            priority = priority_analysis['region_priorities'][region]
            report += f"- **{region}**: {priority['weakest_domain']} 영역 개선 필요 (점수: {priority['weakest_score']}점)\n"
        
        report += f"""
## 💡 정책 제언

### 1. 투자 전략
- **우선순위 기반 차등 투자**: 지역별 우선순위에 따른 차등적 투자
- **매우 높음 우선순위**: 40% 투자 집중
- **높음 우선순위**: 35% 투자
- **보통 우선순위**: 20% 투자
- **낮음 우선순위**: 5% 투자

### 2. 실행 로드맵

#### 1단계 (1-2년): 긴급 불균형 해소
- 대상: 매우 높음 우선순위 지역
- 주요 활동: 긴급 투자 프로그램, 기초 인프라 확충

#### 2단계 (3-5년): 균형 발전 기반 구축
- 대상: 높음 우선순위 지역
- 주요 활동: 지속가능한 발전 모델 구축, 지역 특화 프로그램

#### 3단계 (6-10년): 포용적 성장 달성
- 대상: 전체 지역
- 주요 활동: 균형발전 체계 완성, 지역 간 협력 강화

## 📋 결론 및 제언

새 정부의 국가 균형성장 비전 달성을 위해서는 다음과 같은 종합적 접근이 필요합니다:

1. **과학적 근거 기반 정책 수립**: 통합 균형발전 지수를 활용한 객관적 평가
2. **지역별 맞춤형 접근**: 각 지역의 취약 영역을 고려한 차별화된 정책
3. **단계적 실행**: 우선순위에 따른 체계적이고 단계적인 정책 실행
4. **지속적 모니터링**: 정책 효과의 지속적 평가 및 피드백

이를 통해 지역 간 삶의 질 차이, 기후 문제, 복지 격차 등을 종합적으로 해소하여 진정한 의미의 균형발전을 달성할 수 있을 것입니다.
"""
        
        # 보고서 저장
        with open(f'integrated_balance_report_{year}.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"종합 보고서 생성 완료: integrated_balance_report_{year}.md")
        return report

if __name__ == "__main__":
    analyzer = IntegratedBalanceAnalyzer()
    
    # 2025년 종합 분석 실행
    report = analyzer.generate_report(2025)
    print("분석 완료! 보고서가 생성되었습니다.")
