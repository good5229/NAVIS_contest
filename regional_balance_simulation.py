import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RegionalBalanceSimulation:
    def __init__(self):
        self.setup_visualization()
        self.regions = [
            '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시',
            '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도',
            '충청북도', '충청남도', '전라북도', '전라남도', '경상북도',
            '경상남도', '제주도'
        ]
    
    def setup_visualization(self):
        """시각화 설정"""
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
    
    def load_bds_data(self):
        """BDS 데이터 로드"""
        try:
            bds_data = pd.read_csv('comprehensive_bds_data.csv')
            print(f"BDS 데이터 로드: {len(bds_data)} 레코드")
            return bds_data
        except FileNotFoundError:
            print("BDS 데이터가 없습니다. 샘플 데이터를 생성합니다.")
            return self.create_sample_bds_data()
    
    def create_sample_bds_data(self):
        """샘플 BDS 데이터 생성"""
        np.random.seed(42)
        data = []
        years = range(1997, 2026)
        
        for year in years:
            for region in self.regions:
                # 지역별 기본 BDS 값
                base_bds = {
                    '서울특별시': 5.2, '부산광역시': 4.8, '대구광역시': 4.5, '인천광역시': 4.7, '광주광역시': 4.3,
                    '대전광역시': 4.6, '울산광역시': 4.9, '세종특별자치시': 4.4, '경기도': 5.1, '강원도': 3.8,
                    '충청북도': 4.0, '충청남도': 4.2, '전라북도': 3.9, '전라남도': 3.7, '경상북도': 4.1,
                    '경상남도': 4.3, '제주도': 3.5
                }
                
                base_value = base_bds.get(region, 4.0)
                trend = np.sin((year - 1997) * 0.1) * 0.3
                random_factor = np.random.normal(0, 0.2)
                
                bds_score = max(0, min(10, base_value + trend + random_factor))
                
                data.append({
                    'region': region,
                    'year': year,
                    'bds_score': round(bds_score, 2)
                })
        
        return pd.DataFrame(data)
    
    def create_policy_scenarios(self):
        """정책 시나리오 생성"""
        scenarios = {
            'baseline': {
                'name': '현재 정책 유지',
                'description': '기존 정책을 그대로 유지하는 시나리오',
                'policy_effects': {}
            },
            'balanced_growth': {
                'name': '균형 성장 정책',
                'description': '낮은 BDS 지역에 집중 투자하여 지역 간 격차를 줄이는 정책',
                'policy_effects': {
                    '강원도': 0.3, '전라북도': 0.25, '전라남도': 0.25, '제주도': 0.2
                }
            },
            'metropolitan_focus': {
                'name': '대도시 중심 정책',
                'description': '대도시 지역에 집중 투자하여 경제 성장을 극대화하는 정책',
                'policy_effects': {
                    '서울특별시': 0.2, '부산광역시': 0.2, '대구광역시': 0.15, '인천광역시': 0.15,
                    '경기도': 0.2, '울산광역시': 0.15
                }
            },
            'regional_cluster': {
                'name': '지역 클러스터 정책',
                'description': '지역별 특화 산업 클러스터를 육성하는 정책',
                'policy_effects': {
                    '서울특별시': 0.1, '부산광역시': 0.15, '대구광역시': 0.15, '인천광역시': 0.1,
                    '광주광역시': 0.2, '대전광역시': 0.2, '울산광역시': 0.15, '세종특별자치시': 0.25,
                    '경기도': 0.1, '강원도': 0.25, '충청북도': 0.2, '충청남도': 0.2,
                    '전라북도': 0.25, '전라남도': 0.25, '경상북도': 0.2, '경상남도': 0.2, '제주도': 0.25
                }
            },
            'innovation_driven': {
                'name': '혁신 주도 정책',
                'description': 'R&D 투자와 혁신 생태계 구축에 집중하는 정책',
                'policy_effects': {
                    '서울특별시': 0.15, '부산광역시': 0.2, '대구광역시': 0.15, '인천광역시': 0.2,
                    '광주광역시': 0.25, '대전광역시': 0.25, '울산광역시': 0.2, '세종특별자치시': 0.3,
                    '경기도': 0.15, '강원도': 0.3, '충청북도': 0.25, '충청남도': 0.25,
                    '전라북도': 0.3, '전라남도': 0.3, '경상북도': 0.25, '경상남도': 0.25, '제주도': 0.3
                }
            }
        }
        
        return scenarios
    
    def simulate_policy_effects(self, bds_data, scenarios):
        """정책 효과 시뮬레이션"""
        print("=== 정책 효과 시뮬레이션 시작 ===")
        
        simulation_results = []
        
        for scenario_name, scenario_info in scenarios.items():
            print(f"시뮬레이션 중: {scenario_info['name']}")
            
            # 기본 데이터 복사
            scenario_data = bds_data.copy()
            scenario_data['scenario'] = scenario_name
            scenario_data['policy_name'] = scenario_info['name']
            
            # 정책 효과 적용
            policy_effects = scenario_info.get('policy_effects', {})
            
            for region, effect in policy_effects.items():
                mask = scenario_data['region'] == region
                # 2020년 이후에만 정책 효과 적용
                future_mask = mask & (scenario_data['year'] > 2020)
                
                if future_mask.any():
                    # 점진적 효과 적용 (연도별로 효과 증가)
                    for year in range(2021, 2026):
                        year_mask = future_mask & (scenario_data['year'] == year)
                        if year_mask.any():
                            years_from_2020 = year - 2020
                            cumulative_effect = effect * (years_from_2020 / 5)  # 5년간 점진적 적용
                            scenario_data.loc[year_mask, 'bds_score'] += cumulative_effect
                            scenario_data.loc[year_mask, 'bds_score'] = np.clip(
                                scenario_data.loc[year_mask, 'bds_score'], 0, 10
                            )
            
            simulation_results.append(scenario_data)
        
        return pd.concat(simulation_results, ignore_index=True)
    
    def analyze_regional_balance(self, simulation_data):
        """지역 균형 분석"""
        print("=== 지역 균형 분석 ===")
        
        analysis_results = []
        
        for scenario in simulation_data['scenario'].unique():
            scenario_data = simulation_data[simulation_data['scenario'] == scenario]
            
            # 2025년 데이터로 분석
            final_data = scenario_data[scenario_data['year'] == 2025]
            
            if len(final_data) > 0:
                # 지역 간 격차 분석
                bds_scores = final_data['bds_score']
                regional_gap = bds_scores.max() - bds_scores.min()
                regional_variance = bds_scores.var()
                gini_coefficient = self.calculate_gini(bds_scores)
                
                # 지역 분류
                high_bds = final_data[final_data['bds_score'] >= 4.5]
                medium_bds = final_data[(final_data['bds_score'] >= 3.5) & (final_data['bds_score'] < 4.5)]
                low_bds = final_data[final_data['bds_score'] < 3.5]
                
                analysis_results.append({
                    'scenario': scenario,
                    'policy_name': final_data['policy_name'].iloc[0],
                    'avg_bds': round(bds_scores.mean(), 2),
                    'max_bds': round(bds_scores.max(), 2),
                    'min_bds': round(bds_scores.min(), 2),
                    'regional_gap': round(regional_gap, 2),
                    'regional_variance': round(regional_variance, 3),
                    'gini_coefficient': round(gini_coefficient, 3),
                    'high_bds_count': len(high_bds),
                    'medium_bds_count': len(medium_bds),
                    'low_bds_count': len(low_bds),
                    'balance_score': round(10 - regional_gap, 2)  # 격차가 작을수록 높은 점수
                })
        
        return pd.DataFrame(analysis_results)
    
    def calculate_gini(self, values):
        """지니계수 계산"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def create_visualization(self, simulation_data, analysis_results):
        """시각화 생성"""
        print("=== 시각화 생성 ===")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 시나리오별 BDS 분포 비교 (2025년)
        ax1 = plt.subplot(3, 3, 1)
        final_data = simulation_data[simulation_data['year'] == 2025]
        
        for scenario in final_data['scenario'].unique():
            scenario_data = final_data[final_data['scenario'] == scenario]
            ax1.hist(scenario_data['bds_score'], alpha=0.6, label=scenario_data['policy_name'].iloc[0], bins=10)
        
        ax1.set_title('시나리오별 BDS 분포 (2025년)', fontweight='bold')
        ax1.set_xlabel('BDS 점수')
        ax1.set_ylabel('지역 수')
        ax1.legend()
        
        # 2. 지역 간 격차 비교
        ax2 = plt.subplot(3, 3, 2)
        scenarios = analysis_results['scenario']
        gaps = analysis_results['regional_gap']
        
        bars = ax2.bar(scenarios, gaps, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax2.set_title('지역 간 BDS 격차', fontweight='bold')
        ax2.set_ylabel('격차 (최고 - 최저)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{gap}', ha='center', va='bottom')
        
        # 3. 균형 점수 비교
        ax3 = plt.subplot(3, 3, 3)
        balance_scores = analysis_results['balance_score']
        
        bars = ax3.bar(scenarios, balance_scores, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax3.set_title('지역 균형 점수', fontweight='bold')
        ax3.set_ylabel('균형 점수 (높을수록 균형적)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, score in zip(bars, balance_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{score}', ha='center', va='bottom')
        
        # 4. 지니계수 비교
        ax4 = plt.subplot(3, 3, 4)
        gini_coefficients = analysis_results['gini_coefficient']
        
        bars = ax4.bar(scenarios, gini_coefficients, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax4.set_title('지니계수 (불평등도)', fontweight='bold')
        ax4.set_ylabel('지니계수 (낮을수록 평등)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, gini in zip(bars, gini_coefficients):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{gini}', ha='center', va='bottom')
        
        # 5. 지역별 BDS 변화 (균형 성장 정책)
        ax5 = plt.subplot(3, 3, 5)
        balanced_data = simulation_data[simulation_data['scenario'] == 'balanced_growth']
        
        for region in ['서울특별시', '강원도', '전라남도', '제주도']:
            region_data = balanced_data[balanced_data['region'] == region]
            ax5.plot(region_data['year'], region_data['bds_score'], 
                    marker='o', label=region, linewidth=2)
        
        ax5.set_title('균형 성장 정책 효과', fontweight='bold')
        ax5.set_xlabel('연도')
        ax5.set_ylabel('BDS 점수')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 지역 분류 비교
        ax6 = plt.subplot(3, 3, 6)
        categories = ['높은 BDS', '중간 BDS', '낮은 BDS']
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        high_counts = analysis_results['high_bds_count']
        medium_counts = analysis_results['medium_bds_count']
        low_counts = analysis_results['low_bds_count']
        
        ax6.bar(x - width, high_counts, width, label='높은 BDS (4.5+)', alpha=0.7)
        ax6.bar(x, medium_counts, width, label='중간 BDS (3.5-4.5)', alpha=0.7)
        ax6.bar(x + width, low_counts, width, label='낮은 BDS (<3.5)', alpha=0.7)
        
        ax6.set_title('지역 분류별 개수', fontweight='bold')
        ax6.set_ylabel('지역 수')
        ax6.set_xticks(x)
        ax6.set_xticklabels(scenarios, rotation=45)
        ax6.legend()
        
        # 7. 시나리오별 평균 BDS 비교
        ax7 = plt.subplot(3, 3, 7)
        avg_bds = analysis_results['avg_bds']
        
        bars = ax7.bar(scenarios, avg_bds, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax7.set_title('시나리오별 평균 BDS', fontweight='bold')
        ax7.set_ylabel('평균 BDS 점수')
        ax7.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, avg in zip(bars, avg_bds):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{avg}', ha='center', va='bottom')
        
        # 8. 정책 효과 히트맵
        ax8 = plt.subplot(3, 3, 8)
        final_data_pivot = final_data.pivot(index='region', columns='scenario', values='bds_score')
        
        sns.heatmap(final_data_pivot, annot=True, cmap='RdYlBu_r', center=4.5, ax=ax8, fmt='.1f')
        ax8.set_title('지역별 BDS 점수 (2025년)', fontweight='bold')
        
        # 9. 균형 vs 성장 트레이드오프
        ax9 = plt.subplot(3, 3, 9)
        ax9.scatter(analysis_results['avg_bds'], analysis_results['balance_score'], 
                   s=100, alpha=0.7, c=['blue', 'green', 'red', 'orange', 'purple'])
        
        for i, scenario in enumerate(scenarios):
            ax9.annotate(analysis_results['policy_name'].iloc[i], 
                        (analysis_results['avg_bds'].iloc[i], analysis_results['balance_score'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax9.set_title('균형 vs 성장 트레이드오프', fontweight='bold')
        ax9.set_xlabel('평균 BDS (성장)')
        ax9.set_ylabel('균형 점수 (균형)')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regional_balance_simulation.png', dpi=300, bbox_inches='tight')
        print("시각화가 regional_balance_simulation.png에 저장되었습니다.")
        
        return fig
    
    def generate_report(self, analysis_results, scenarios):
        """분석 보고서 생성"""
        print("=== 분석 보고서 생성 ===")
        
        # 최적 시나리오 찾기
        best_balance = analysis_results.loc[analysis_results['balance_score'].idxmax()]
        best_growth = analysis_results.loc[analysis_results['avg_bds'].idxmax()]
        best_overall = analysis_results.loc[(analysis_results['avg_bds'] * analysis_results['balance_score']).idxmax()]
        
        report_content = f"""# 지역균형발전 시뮬레이션 분석 보고서

## 📊 분석 개요
- **분석 기간**: 1997-2025년
- **분석 지역**: 17개 광역시 및 도
- **시뮬레이션 시나리오**: 5개 정책 시나리오
- **생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 정책 시나리오

"""
        
        for scenario_name, scenario_info in scenarios.items():
            report_content += f"""
### {scenario_info['name']}
- **설명**: {scenario_info['description']}
- **주요 효과**: {', '.join([f'{region}(+{effect})' for region, effect in scenario_info.get('policy_effects', {}).items()]) if scenario_info.get('policy_effects') else '효과 없음'}

"""
        
        report_content += f"""
## 📈 분석 결과

### 최적 시나리오
- **최고 균형 점수**: {best_balance['policy_name']} ({best_balance['balance_score']}점)
- **최고 성장 점수**: {best_growth['policy_name']} ({best_growth['avg_bds']}점)
- **최고 종합 점수**: {best_overall['policy_name']} (균형: {best_overall['balance_score']}점, 성장: {best_overall['avg_bds']}점)

### 시나리오별 상세 분석

"""
        
        for _, row in analysis_results.iterrows():
            report_content += f"""
#### {row['policy_name']}
- **평균 BDS**: {row['avg_bds']}점
- **지역 간 격차**: {row['regional_gap']}점
- **균형 점수**: {row['balance_score']}점
- **지니계수**: {row['gini_coefficient']}
- **지역 분류**: 높은 BDS {row['high_bds_count']}개, 중간 BDS {row['medium_bds_count']}개, 낮은 BDS {row['low_bds_count']}개

"""
        
        report_content += f"""
## 🎯 정책 권고사항

### 1. 균형 성장 정책 추천
- **적용 대상**: {best_balance['policy_name']}
- **기대 효과**: 지역 간 격차 최소화
- **적용 방안**: 낮은 BDS 지역에 집중 투자

### 2. 성장 중심 정책 추천
- **적용 대상**: {best_growth['policy_name']}
- **기대 효과**: 전체적인 경제 성장 극대화
- **적용 방안**: 대도시 중심 투자

### 3. 종합적 접근 정책 추천
- **적용 대상**: {best_overall['policy_name']}
- **기대 효과**: 성장과 균형의 최적 조합
- **적용 방안**: 지역 특성에 맞는 맞춤형 정책

## 📊 시각화 파일
- **시뮬레이션 결과**: regional_balance_simulation.png
- **분석 데이터**: regional_balance_analysis.csv

---
*이 보고서는 BDS 데이터를 기반으로 한 지역균형발전 시뮬레이션 결과입니다.*
"""
        
        # 보고서 저장
        with open('regional_balance_simulation_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("분석 보고서 생성 완료: regional_balance_simulation_report.md")
        
        return report_content

def main():
    simulator = RegionalBalanceSimulation()
    
    print("=== 지역균형발전 시뮬레이션 시작 ===")
    
    # 1. BDS 데이터 로드
    print("1. BDS 데이터 로드 중...")
    bds_data = simulator.load_bds_data()
    
    # 2. 정책 시나리오 생성
    print("2. 정책 시나리오 생성 중...")
    scenarios = simulator.create_policy_scenarios()
    
    # 3. 정책 효과 시뮬레이션
    print("3. 정책 효과 시뮬레이션 중...")
    simulation_data = simulator.simulate_policy_effects(bds_data, scenarios)
    
    # 4. 지역 균형 분석
    print("4. 지역 균형 분석 중...")
    analysis_results = simulator.analyze_regional_balance(simulation_data)
    
    # 5. 시각화 생성
    print("5. 시각화 생성 중...")
    fig = simulator.create_visualization(simulation_data, analysis_results)
    
    # 6. 보고서 생성
    print("6. 분석 보고서 생성 중...")
    report = simulator.generate_report(analysis_results, scenarios)
    
    # 7. 결과 저장
    simulation_data.to_csv('regional_balance_simulation_data.csv', index=False, encoding='utf-8-sig')
    analysis_results.to_csv('regional_balance_analysis.csv', index=False, encoding='utf-8-sig')
    
    print("\n=== 시뮬레이션 완료 ===")
    print("- 시각화: regional_balance_simulation.png")
    print("- 보고서: regional_balance_simulation_report.md")
    print("- 시뮬레이션 데이터: regional_balance_simulation_data.csv")
    print("- 분석 결과: regional_balance_analysis.csv")

if __name__ == "__main__":
    main()
