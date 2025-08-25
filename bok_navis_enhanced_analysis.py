import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sqlite3
import json
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class BOKNAVISEnhancedAnalysis:
    def __init__(self, api_key='XQL2HWK4J7RF995OEDNG'):
        self.api_key = api_key
        self.base_url = 'https://ecos.bok.or.kr/api'
        self.setup_visualization()
    
    def setup_visualization(self):
        """시각화 설정"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    def get_current_economic_indicators(self):
        """현재 경제 지표 조회"""
        url = f'{self.base_url}/KeyStatisticList/{self.api_key}/json/kr/1/100'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'KeyStatisticList' in data and 'row' in data['KeyStatisticList']:
                df = pd.json_normalize(data['KeyStatisticList']['row'])
                return df
            else:
                return None
                
        except Exception as e:
            print(f"API 요청 오류: {e}")
            return None
    
    def create_economic_dashboard_data(self, df):
        """경제 대시보드용 데이터 생성"""
        if df is None:
            return None
        
        # 시장금리 데이터
        market_rates = df[df['CLASS_NAME'] == '시장금리'].copy()
        market_rates['DATA_VALUE'] = pd.to_numeric(market_rates['DATA_VALUE'], errors='coerce')
        
        # 환율 데이터
        exchange_rates = df[df['CLASS_NAME'] == '환율'].copy()
        exchange_rates['DATA_VALUE'] = pd.to_numeric(exchange_rates['DATA_VALUE'], errors='coerce')
        
        # 물가 데이터
        price_indicators = df[df['CLASS_NAME'].str.contains('물가', na=False)].copy()
        price_indicators['DATA_VALUE'] = pd.to_numeric(price_indicators['DATA_VALUE'], errors='coerce')
        
        # 성장률 데이터
        growth_indicators = df[df['CLASS_NAME'] == '성장률'].copy()
        growth_indicators['DATA_VALUE'] = pd.to_numeric(growth_indicators['DATA_VALUE'], errors='coerce')
        
        return {
            'market_rates': market_rates,
            'exchange_rates': exchange_rates,
            'price_indicators': price_indicators,
            'growth_indicators': growth_indicators
        }
    
    def load_existing_navis_data(self):
        """기존 NAVIS 프로젝트 데이터 로드"""
        try:
            # 기존 BDS 모델 데이터 로드
            bds_data = pd.read_csv('enhanced_bds_model_with_kosis.csv')
            print(f"BDS 모델 데이터 로드: {len(bds_data)}개 행")
            return bds_data
        except FileNotFoundError:
            print("기존 BDS 모델 데이터를 찾을 수 없습니다.")
            return None
    
    def create_synthetic_economic_scenarios(self, current_indicators):
        """현재 경제 지표를 기반으로 시나리오 생성"""
        print("=== 경제 시나리오 생성 ===")
        
        # 기준 시나리오 (현재 상황)
        base_scenario = {
            'base_rate': 2.5,  # 기준금리
            'exchange_rate': 1393.2,  # 원/달러 환율
            'cpi': 116.52,  # 소비자물가지수
            'gdp_growth': 0.6,  # GDP 성장률
            'koribor': 2.49,  # KORIBOR
            'gov_bond_3y': 2.456,  # 국고채 3년
            'corp_bond_3y': 2.932,  # 회사채 3년
        }
        
        # 시나리오 변형
        scenarios = {
            'baseline': base_scenario,
            'optimistic': {
                'base_rate': 2.0,
                'exchange_rate': 1300.0,
                'cpi': 115.0,
                'gdp_growth': 1.2,
                'koribor': 2.0,
                'gov_bond_3y': 2.0,
                'corp_bond_3y': 2.5,
            },
            'pessimistic': {
                'base_rate': 3.5,
                'exchange_rate': 1500.0,
                'cpi': 120.0,
                'gdp_growth': 0.0,
                'koribor': 3.5,
                'gov_bond_3y': 3.5,
                'corp_bond_3y': 4.0,
            },
            'high_inflation': {
                'base_rate': 4.0,
                'exchange_rate': 1450.0,
                'cpi': 125.0,
                'gdp_growth': 0.3,
                'koribor': 4.0,
                'gov_bond_3y': 4.0,
                'corp_bond_3y': 4.5,
            },
            'low_growth': {
                'base_rate': 1.5,
                'exchange_rate': 1350.0,
                'cpi': 114.0,
                'gdp_growth': -0.5,
                'koribor': 1.5,
                'gov_bond_3y': 1.5,
                'corp_bond_3y': 2.0,
            }
        }
        
        return scenarios
    
    def enhance_bds_model_with_economic_data(self, bds_data, scenarios):
        """BDS 모델을 경제 데이터로 강화"""
        print("=== BDS 모델 경제 데이터 강화 ===")
        
        if bds_data is None:
            print("BDS 데이터가 없어 시뮬레이션 데이터를 생성합니다.")
            # 시뮬레이션 데이터 생성
            np.random.seed(42)
            n_periods = 100
            n_scenarios = len(scenarios)
            
            enhanced_data = []
            
            for scenario_name, scenario_params in scenarios.items():
                for period in range(n_periods):
                    # 경제 지표에 노이즈 추가
                    noise_factor = 0.1
                    
                    row = {
                        'scenario': scenario_name,
                        'period': period,
                        'base_rate': scenario_params['base_rate'] + np.random.normal(0, noise_factor),
                        'exchange_rate': scenario_params['exchange_rate'] + np.random.normal(0, noise_factor * 10),
                        'cpi': scenario_params['cpi'] + np.random.normal(0, noise_factor),
                        'gdp_growth': scenario_params['gdp_growth'] + np.random.normal(0, noise_factor),
                        'koribor': scenario_params['koribor'] + np.random.normal(0, noise_factor),
                        'gov_bond_3y': scenario_params['gov_bond_3y'] + np.random.normal(0, noise_factor),
                        'corp_bond_3y': scenario_params['corp_bond_3y'] + np.random.normal(0, noise_factor),
                    }
                    
                    # BDS 모델 변수들 (시뮬레이션)
                    row['bds_score'] = self.calculate_bds_score(row)
                    row['economic_stress'] = self.calculate_economic_stress(row)
                    row['financial_stability'] = self.calculate_financial_stability(row)
                    row['growth_potential'] = self.calculate_growth_potential(row)
                    
                    enhanced_data.append(row)
            
            enhanced_df = pd.DataFrame(enhanced_data)
        else:
            # 기존 BDS 데이터에 경제 지표 추가
            enhanced_df = bds_data.copy()
            
            # 시나리오별로 데이터 확장
            scenario_data = []
            for scenario_name, scenario_params in scenarios.items():
                scenario_df = enhanced_df.copy()
                scenario_df['scenario'] = scenario_name
                
                # 경제 지표 추가
                for key, value in scenario_params.items():
                    if key not in scenario_df.columns:
                        scenario_df[key] = value
                
                scenario_data.append(scenario_df)
            
            enhanced_df = pd.concat(scenario_data, ignore_index=True)
        
        return enhanced_df
    
    def calculate_bds_score(self, economic_data):
        """경제 지표를 기반으로 BDS 점수 계산"""
        # 가중 평균 방식으로 BDS 점수 계산
        weights = {
            'base_rate': -0.2,  # 기준금리 상승은 부정적
            'exchange_rate': -0.15,  # 환율 상승은 부정적
            'cpi': -0.25,  # 물가상승은 부정적
            'gdp_growth': 0.3,  # 성장률은 긍정적
            'koribor': -0.1,  # 금리 상승은 부정적
        }
        
        score = 0
        for indicator, weight in weights.items():
            if indicator in economic_data:
                # 정규화 (0-100 스케일)
                normalized_value = min(max(economic_data[indicator] / 10, 0), 10)
                score += weight * normalized_value
        
        return max(0, min(100, 50 + score * 10))  # 0-100 범위로 제한
    
    def calculate_economic_stress(self, economic_data):
        """경제 스트레스 지수 계산"""
        stress_factors = []
        
        if 'base_rate' in economic_data:
            stress_factors.append(economic_data['base_rate'] / 5.0)  # 기준금리 스트레스
        
        if 'exchange_rate' in economic_data:
            stress_factors.append((economic_data['exchange_rate'] - 1200) / 300)  # 환율 스트레스
        
        if 'cpi' in economic_data:
            stress_factors.append((economic_data['cpi'] - 100) / 20)  # 물가 스트레스
        
        if 'gdp_growth' in economic_data:
            stress_factors.append(max(0, -economic_data['gdp_growth']))  # 성장률 스트레스
        
        return min(100, np.mean(stress_factors) * 100) if stress_factors else 0
    
    def calculate_financial_stability(self, economic_data):
        """금융 안정성 지수 계산"""
        stability_factors = []
        
        if 'gov_bond_3y' in economic_data and 'corp_bond_3y' in economic_data:
            # 신용 스프레드 (낮을수록 안정적)
            credit_spread = economic_data['corp_bond_3y'] - economic_data['gov_bond_3y']
            stability_factors.append(max(0, 100 - credit_spread * 20))
        
        if 'base_rate' in economic_data:
            # 금리 안정성 (적정 수준에서 벗어날수록 불안정)
            rate_stability = max(0, 100 - abs(economic_data['base_rate'] - 2.5) * 20)
            stability_factors.append(rate_stability)
        
        if 'exchange_rate' in economic_data:
            # 환율 안정성
            exchange_stability = max(0, 100 - abs(economic_data['exchange_rate'] - 1300) / 10)
            stability_factors.append(exchange_stability)
        
        return np.mean(stability_factors) if stability_factors else 50
    
    def calculate_growth_potential(self, economic_data):
        """성장 잠재력 지수 계산"""
        growth_factors = []
        
        if 'gdp_growth' in economic_data:
            growth_factors.append(max(0, economic_data['gdp_growth'] * 50 + 50))
        
        if 'base_rate' in economic_data:
            # 적정 금리 범위에서의 성장 잠재력
            rate_factor = max(0, 100 - abs(economic_data['base_rate'] - 2.5) * 15)
            growth_factors.append(rate_factor)
        
        if 'cpi' in economic_data:
            # 적정 물가 수준에서의 성장 잠재력
            inflation_factor = max(0, 100 - abs(economic_data['cpi'] - 100) * 2)
            growth_factors.append(inflation_factor)
        
        return np.mean(growth_factors) if growth_factors else 50
    
    def perform_scenario_analysis(self, enhanced_data):
        """시나리오 분석 수행"""
        print("=== 시나리오 분석 수행 ===")
        
        # 시나리오별 평균 지표 계산
        scenario_summary = enhanced_data.groupby('scenario').agg({
            'bds_score': ['mean', 'std'],
            'economic_stress': ['mean', 'std'],
            'financial_stability': ['mean', 'std'],
            'growth_potential': ['mean', 'std'],
            'base_rate': 'mean',
            'exchange_rate': 'mean',
            'cpi': 'mean',
            'gdp_growth': 'mean'
        }).round(2)
        
        print("\n시나리오별 평균 지표:")
        print(scenario_summary)
        
        return scenario_summary
    
    def create_comprehensive_visualization(self, enhanced_data, scenario_summary):
        """종합 시각화 생성"""
        print("=== 종합 시각화 생성 ===")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 시나리오별 BDS 점수 비교
        ax1 = plt.subplot(3, 3, 1)
        bds_means = scenario_summary[('bds_score', 'mean')]
        bds_stds = scenario_summary[('bds_score', 'std')]
        
        bars = ax1.bar(bds_means.index, bds_means.values, yerr=bds_stds.values, 
                      capsize=5, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax1.set_title('BDS Score by Scenario', fontweight='bold')
        ax1.set_ylabel('BDS Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, mean_val in zip(bars, bds_means.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean_val:.1f}', ha='center', va='bottom')
        
        # 2. 경제 스트레스 지수
        ax2 = plt.subplot(3, 3, 2)
        stress_means = scenario_summary[('economic_stress', 'mean')]
        stress_stds = scenario_summary[('economic_stress', 'std')]
        
        bars = ax2.bar(stress_means.index, stress_means.values, yerr=stress_stds.values,
                      capsize=5, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax2.set_title('Economic Stress Index', fontweight='bold')
        ax2.set_ylabel('Stress Level')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 금융 안정성
        ax3 = plt.subplot(3, 3, 3)
        stability_means = scenario_summary[('financial_stability', 'mean')]
        stability_stds = scenario_summary[('financial_stability', 'std')]
        
        bars = ax3.bar(stability_means.index, stability_means.values, yerr=stability_stds.values,
                      capsize=5, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax3.set_title('Financial Stability Index', fontweight='bold')
        ax3.set_ylabel('Stability Level')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 성장 잠재력
        ax4 = plt.subplot(3, 3, 4)
        growth_means = scenario_summary[('growth_potential', 'mean')]
        growth_stds = scenario_summary[('growth_potential', 'std')]
        
        bars = ax4.bar(growth_means.index, growth_means.values, yerr=growth_stds.values,
                      capsize=5, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax4.set_title('Growth Potential Index', fontweight='bold')
        ax4.set_ylabel('Growth Potential')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 경제 지표 비교 (기준금리)
        ax5 = plt.subplot(3, 3, 5)
        base_rates = scenario_summary[('base_rate', 'mean')]
        bars = ax5.bar(base_rates.index, base_rates.values, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax5.set_title('Base Interest Rate by Scenario', fontweight='bold')
        ax5.set_ylabel('Rate (%)')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 환율 비교
        ax6 = plt.subplot(3, 3, 6)
        exchange_rates = scenario_summary[('exchange_rate', 'mean')]
        bars = ax6.bar(exchange_rates.index, exchange_rates.values, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax6.set_title('Exchange Rate (KRW/USD)', fontweight='bold')
        ax6.set_ylabel('Rate (KRW)')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. 시나리오별 지표 상관관계 히트맵
        ax7 = plt.subplot(3, 3, 7)
        correlation_data = enhanced_data[['bds_score', 'economic_stress', 'financial_stability', 'growth_potential']].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=ax7)
        ax7.set_title('Indicator Correlations', fontweight='bold')
        
        # 8. 시나리오별 BDS 점수 분포
        ax8 = plt.subplot(3, 3, 8)
        for scenario in enhanced_data['scenario'].unique():
            scenario_data = enhanced_data[enhanced_data['scenario'] == scenario]['bds_score']
            ax8.hist(scenario_data, alpha=0.5, label=scenario, bins=20)
        ax8.set_title('BDS Score Distribution by Scenario', fontweight='bold')
        ax8.set_xlabel('BDS Score')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        
        # 9. 경제 지표 vs BDS 점수 산점도
        ax9 = plt.subplot(3, 3, 9)
        ax9.scatter(enhanced_data['base_rate'], enhanced_data['bds_score'], 
                   c=enhanced_data['economic_stress'], cmap='viridis', alpha=0.6)
        ax9.set_title('Base Rate vs BDS Score', fontweight='bold')
        ax9.set_xlabel('Base Rate (%)')
        ax9.set_ylabel('BDS Score')
        
        plt.tight_layout()
        plt.savefig('bok_navis_enhanced_analysis.png', dpi=300, bbox_inches='tight')
        print("종합 분석 시각화가 bok_navis_enhanced_analysis.png에 저장되었습니다.")
        
        return fig
    
    def generate_enhanced_report(self, enhanced_data, scenario_summary, current_indicators):
        """향상된 분석 보고서 생성"""
        print("=== 향상된 분석 보고서 생성 ===")
        
        report = []
        report.append("# 한국은행 API 기반 NAVIS 향상된 분석 보고서")
        report.append(f"생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}")
        report.append("")
        
        # 현재 경제 상황 요약
        if current_indicators is not None:
            report.append("## 1. 현재 경제 상황 요약")
            report.append("")
            
            # 기준금리
            base_rate = current_indicators[current_indicators['KEYSTAT_NAME'] == '한국은행 기준금리']
            if not base_rate.empty:
                report.append(f"- **한국은행 기준금리**: {base_rate.iloc[0]['DATA_VALUE']}%")
            
            # 환율
            usd_rate = current_indicators[current_indicators['KEYSTAT_NAME'].str.contains('원/달러', na=False)]
            if not usd_rate.empty:
                report.append(f"- **원/달러 환율**: {usd_rate.iloc[0]['DATA_VALUE']}원")
            
            # GDP 성장률
            gdp_growth = current_indicators[current_indicators['KEYSTAT_NAME'].str.contains('경제성장률', na=False)]
            if not gdp_growth.empty:
                report.append(f"- **실질GDP 성장률**: {gdp_growth.iloc[0]['DATA_VALUE']}%")
            
            # CPI
            cpi = current_indicators[current_indicators['KEYSTAT_NAME'] == '소비자물가지수']
            if not cpi.empty:
                report.append(f"- **소비자물가지수**: {cpi.iloc[0]['DATA_VALUE']}")
            
            report.append("")
        
        # 시나리오 분석 결과
        report.append("## 2. 시나리오 분석 결과")
        report.append("")
        
        # 최적 시나리오 찾기
        best_bds_scenario = scenario_summary[('bds_score', 'mean')].idxmax()
        best_stability_scenario = scenario_summary[('financial_stability', 'mean')].idxmax()
        best_growth_scenario = scenario_summary[('growth_potential', 'mean')].idxmax()
        
        report.append(f"- **최고 BDS 점수 시나리오**: {best_bds_scenario} ({scenario_summary.loc[best_bds_scenario, ('bds_score', 'mean')]:.1f}점)")
        report.append(f"- **최고 금융안정성 시나리오**: {best_stability_scenario} ({scenario_summary.loc[best_stability_scenario, ('financial_stability', 'mean')]:.1f}점)")
        report.append(f"- **최고 성장잠재력 시나리오**: {best_growth_scenario} ({scenario_summary.loc[best_growth_scenario, ('growth_potential', 'mean')]:.1f}점)")
        report.append("")
        
        # 시나리오별 상세 분석
        report.append("### 시나리오별 상세 분석")
        report.append("")
        
        for scenario in scenario_summary.index:
            report.append(f"#### {scenario} 시나리오")
            report.append("")
            
            bds_score = scenario_summary.loc[scenario, ('bds_score', 'mean')]
            stress = scenario_summary.loc[scenario, ('economic_stress', 'mean')]
            stability = scenario_summary.loc[scenario, ('financial_stability', 'mean')]
            growth = scenario_summary.loc[scenario, ('growth_potential', 'mean')]
            
            report.append(f"- **BDS 점수**: {bds_score:.1f}점")
            report.append(f"- **경제 스트레스**: {stress:.1f}점")
            report.append(f"- **금융 안정성**: {stability:.1f}점")
            report.append(f"- **성장 잠재력**: {growth:.1f}점")
            report.append("")
        
        # 정책 권고사항
        report.append("## 3. 정책 권고사항")
        report.append("")
        
        # 현재 상황 분석
        current_bds = scenario_summary.loc['baseline', ('bds_score', 'mean')]
        current_stress = scenario_summary.loc['baseline', ('economic_stress', 'mean')]
        
        if current_bds < 50:
            report.append("- **BDS 점수가 낮음**: 경제 활성화 정책 필요")
        else:
            report.append("- **BDS 점수가 양호**: 현재 정책 유지 권고")
        
        if current_stress > 50:
            report.append("- **경제 스트레스 높음**: 안정화 정책 우선")
        else:
            report.append("- **경제 스트레스 양호**: 성장 정책 추진 가능")
        
        report.append("")
        
        # 향후 전망
        report.append("## 4. 향후 전망")
        report.append("")
        
        # 가장 가능성이 높은 시나리오
        most_likely = 'baseline'
        if current_indicators is not None:
            base_rate_val = float(base_rate.iloc[0]['DATA_VALUE']) if not base_rate.empty else 2.5
            if base_rate_val > 3.0:
                most_likely = 'pessimistic'
            elif base_rate_val < 2.0:
                most_likely = 'optimistic'
        
        report.append(f"- **가장 가능성이 높은 시나리오**: {most_likely}")
        report.append(f"- **예상 BDS 점수**: {scenario_summary.loc[most_likely, ('bds_score', 'mean')]:.1f}점")
        report.append(f"- **예상 경제 스트레스**: {scenario_summary.loc[most_likely, ('economic_stress', 'mean')]:.1f}점")
        report.append("")
        
        # 보고서 저장
        with open('bok_navis_enhanced_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("향상된 분석 보고서가 bok_navis_enhanced_report.md에 저장되었습니다.")
        return report

    def analyze_correlation_with_navis(self):
        """NAVIS 데이터와 새로운 BDS의 상관관계 분석"""
        print("=== NAVIS와 새로운 BDS 상관관계 분석 ===")
        
        # 새로운 BDS 데이터 로드
        try:
            comprehensive_bds = pd.read_csv('comprehensive_bds_data.csv')
            print(f"종합 BDS 데이터 로드: {len(comprehensive_bds)} 레코드")
        except FileNotFoundError:
            print("종합 BDS 데이터가 없습니다.")
            return None
        
        # NAVIS 데이터 로드
        try:
            navis_data = pd.read_csv('enhanced_bds_model_with_kosis.csv')
            print(f"NAVIS 데이터 로드: {len(navis_data)} 레코드")
        except FileNotFoundError:
            print("NAVIS 데이터가 없습니다.")
            return None
        
        # 데이터 병합 (지역과 연도 기준)
        merged_data = pd.merge(
            comprehensive_bds, 
            navis_data[['region', 'year', 'bds_value']], 
            on=['region', 'year'], 
            how='inner',
            suffixes=('_new', '_navis')
        )
        
        if merged_data.empty:
            print("병합할 데이터가 없습니다.")
            return None
        
        # 상관관계 분석
        correlation = merged_data['bds_score'].corr(merged_data['bds_value'])
        
        # 지역별 상관관계
        region_correlations = []
        for region in merged_data['region'].unique():
            region_data = merged_data[merged_data['region'] == region]
            if len(region_data) > 1:
                corr = region_data['bds_score'].corr(region_data['bds_value'])
                region_correlations.append({
                    'region': region,
                    'correlation': round(corr, 3),
                    'data_points': len(region_data)
                })
        
        # 연도별 상관관계
        year_correlations = []
        for year in merged_data['year'].unique():
            year_data = merged_data[merged_data['year'] == year]
            if len(year_data) > 1:
                corr = year_data['bds_score'].corr(year_data['bds_value'])
                year_correlations.append({
                    'year': year,
                    'correlation': round(corr, 3),
                    'data_points': len(year_data)
                })
        
        # 결과 저장
        correlation_results = {
            'overall_correlation': round(correlation, 3),
            'region_correlations': pd.DataFrame(region_correlations),
            'year_correlations': pd.DataFrame(year_correlations),
            'merged_data': merged_data
        }
        
        # CSV로 저장
        correlation_results['region_correlations'].to_csv('navis_bds_correlation_by_region.csv', index=False)
        correlation_results['year_correlations'].to_csv('navis_bds_correlation_by_year.csv', index=False)
        correlation_results['merged_data'].to_csv('navis_bds_merged_data.csv', index=False)
        
        print(f"전체 상관관계: {correlation_results['overall_correlation']}")
        print(f"지역별 상관관계 분석 완료: {len(correlation_results['region_correlations'])}개 지역")
        print(f"연도별 상관관계 분석 완료: {len(correlation_results['year_correlations'])}개 연도")
        
        return correlation_results

def main():
    analyzer = BOKNAVISEnhancedAnalysis()
    
    print("=== 한국은행 API 기반 NAVIS 향상된 분석 시작 ===")
    
    # 1. 현재 경제 지표 조회
    print("1. 현재 경제 지표 조회 중...")
    current_indicators = analyzer.get_current_economic_indicators()
    
    # 2. 기존 NAVIS 데이터 로드
    print("2. 기존 NAVIS 데이터 로드 중...")
    bds_data = analyzer.load_existing_navis_data()
    
    # 3. 경제 시나리오 생성
    print("3. 경제 시나리오 생성 중...")
    scenarios = analyzer.create_synthetic_economic_scenarios(current_indicators)
    
    # 4. BDS 모델 강화
    print("4. BDS 모델 경제 데이터 강화 중...")
    enhanced_data = analyzer.enhance_bds_model_with_economic_data(bds_data, scenarios)
    
    # 5. 시나리오 분석
    print("5. 시나리오 분석 수행 중...")
    scenario_summary = analyzer.perform_scenario_analysis(enhanced_data)
    
    # 6. 종합 시각화
    print("6. 종합 시각화 생성 중...")
    fig = analyzer.create_comprehensive_visualization(enhanced_data, scenario_summary)
    
    # 7. NAVIS와 새로운 BDS 상관관계 분석
    print("7. NAVIS와 새로운 BDS 상관관계 분석 중...")
    correlation_results = analyzer.analyze_correlation_with_navis()
    
    # 8. 향상된 보고서 생성
    print("8. 향상된 분석 보고서 생성 중...")
    report = analyzer.generate_enhanced_report(enhanced_data, scenario_summary, current_indicators)
    
    # 9. 결과 저장
    enhanced_data.to_csv('bok_navis_enhanced_data.csv', index=False, encoding='utf-8-sig')
    scenario_summary.to_csv('bok_navis_scenario_summary.csv', encoding='utf-8-sig')
    
    print("\n=== 분석 완료 ===")
    print("- 시각화: bok_navis_enhanced_analysis.png")
    print("- 보고서: bok_navis_enhanced_report.md")
    print("- 데이터: bok_navis_enhanced_data.csv")
    print("- 요약: bok_navis_scenario_summary.csv")
    if correlation_results:
        print("- 상관관계 분석: navis_bds_correlation_by_region.csv, navis_bds_correlation_by_year.csv")

if __name__ == "__main__":
    main()
