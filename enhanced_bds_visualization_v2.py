#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 BDS 모델 검증 및 시각화 v2.0

개선사항:
1. 상세한 툴팁 및 설명 추가
2. 데이터 기간 및 개수 확인
3. 전 지역 시계열 비교
4. Geojson 기반 지역별 히트맵
5. 정책 시뮬레이션 기능
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_enhanced_data():
    """향상된 BDS 모델 데이터 로드"""
    try:
        bds_df = pd.read_csv('enhanced_bds_model.csv', encoding='utf-8-sig')
        validation_df = pd.read_csv('enhanced_bds_validation.csv', encoding='utf-8-sig')
        
        print(f"✅ 향상된 BDS 데이터 로드 완료")
        print(f"📊 BDS 모델: {bds_df.shape}")
        print(f"📊 검증 결과: {validation_df.shape}")
        
        # 데이터 기간 확인
        print(f"\n📅 데이터 기간 분석:")
        print(f"  - 연도 범위: {bds_df['year'].min()}~{bds_df['year'].max()}")
        print(f"  - 총 연도 수: {bds_df['year'].nunique()}")
        print(f"  - 지역 수: {bds_df['region'].nunique()}")
        print(f"  - 지역별 데이터 개수: {len(bds_df) // bds_df['region'].nunique()}")
        
        # 지역별 데이터 개수 확인
        region_counts = bds_df.groupby('region').size()
        print(f"  - 지역별 데이터 개수 분포:")
        print(f"    최소: {region_counts.min()}, 최대: {region_counts.max()}")
        
        return bds_df, validation_df
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None, None

def load_korea_geojson():
    """한국 지도 Geojson 로드"""
    try:
        with open('skorea-provinces-2018-geo.json', 'r', encoding='utf-8') as f:
            geojson = json.load(f)
        print("✅ 한국 지도 Geojson 로드 완료")
        return geojson
    except Exception as e:
        print(f"❌ Geojson 로드 실패: {e}")
        return None

def validate_enhanced_model_comprehensive(bds_df, validation_df):
    """향상된 모델 종합 검증 (상세 설명 포함)"""
    print("\n=== 향상된 모델 종합 검증 ===")
    
    # 1. 기본 검증 통계
    total_regions = len(validation_df)
    leading_regions = validation_df['is_leading'].sum()
    independent_regions = validation_df['is_independent'].sum()
    avg_correlation = validation_df['correlation'].mean()
    avg_independence = validation_df['independence_score'].mean()
    avg_volatility_ratio = validation_df['volatility_ratio'].mean()
    
    print(f"📊 기본 검증 결과:")
    print(f"  - 총 지역: {total_regions}개")
    print(f"  - 선행성 우위: {leading_regions}개 ({leading_regions/total_regions*100:.1f}%)")
    print(f"  - 독립성 우위: {independent_regions}개 ({independent_regions/total_regions*100:.1f}%)")
    print(f"  - 평균 상관관계: {avg_correlation:.3f}")
    print(f"  - 평균 독립성: {avg_independence:.3f}")
    print(f"  - 평균 변동성 비율: {avg_volatility_ratio:.3f}")
    
    # 2. 지역별 상세 분석
    print(f"\n🏆 선행성 우위 지역 (Top 5):")
    leading_top5 = validation_df[validation_df['is_leading']].nlargest(5, 'volatility_ratio')
    for _, row in leading_top5.iterrows():
        print(f"  - {row['region']}: 변동성 비율 {row['volatility_ratio']:.3f}")
    
    # 3. 상관관계 분석
    high_corr_regions = validation_df[validation_df['correlation'] > 0.9]
    medium_corr_regions = validation_df[(validation_df['correlation'] > 0.7) & (validation_df['correlation'] <= 0.9)]
    low_corr_regions = validation_df[validation_df['correlation'] <= 0.7]
    
    print(f"\n📈 상관관계 분포:")
    print(f"  - 높은 상관관계 (>0.9): {len(high_corr_regions)}개 지역")
    print(f"  - 중간 상관관계 (0.7-0.9): {len(medium_corr_regions)}개 지역")
    print(f"  - 낮은 상관관계 (≤0.7): {len(low_corr_regions)}개 지역")
    
    # 4. 검증 점수 계산 (상세 설명)
    validation_score = (
        (leading_regions / total_regions) * 0.4 +  # 선행성 가중치 40%
        (avg_independence) * 0.3 +  # 독립성 가중치 30%
        (avg_volatility_ratio - 1) * 0.3  # 변동성 가중치 30%
    )
    
    print(f"\n🏅 종합 검증 점수: {validation_score:.3f}")
    print(f"   📝 점수 구성:")
    print(f"     - 선행성 점수: {(leading_regions / total_regions) * 0.4:.3f} (40% 가중치)")
    print(f"     - 독립성 점수: {avg_independence * 0.3:.3f} (30% 가중치)")
    print(f"     - 변동성 점수: {(avg_volatility_ratio - 1) * 0.3:.3f} (30% 가중치)")
    
    return {
        'total_regions': total_regions,
        'leading_regions': leading_regions,
        'independent_regions': independent_regions,
        'avg_correlation': avg_correlation,
        'avg_independence': avg_independence,
        'avg_volatility_ratio': avg_volatility_ratio,
        'validation_score': validation_score,
        'leading_top5': leading_top5,
        'correlation_distribution': {
            'high': len(high_corr_regions),
            'medium': len(medium_corr_regions),
            'low': len(low_corr_regions)
        }
    }

def create_comprehensive_visualization_v2(bds_df, validation_df, validation_results, geojson):
    """종합 시각화 생성 v2.0 (상세 툴팁 포함)"""
    print("\n=== 종합 시각화 생성 v2.0 ===")
    
    # 1. 메인 대시보드 (HTML)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'NAVIS vs BDS 상관관계 분포',
            '선행성 우위 지역 분석',
            '지역별 상관관계 히트맵',
            '변동성 비율 분석',
            '전 지역 시계열 비교 (1/2)',
            '전 지역 시계열 비교 (2/2)',
            '검증 점수 요약',
            '지역별 성능 등급'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "choropleth"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "indicator"}, {"type": "bar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1-1. 상관관계 분포 (상세 툴팁)
    correlation_ranges = ['높음 (>0.9)', '중간 (0.7-0.9)', '낮음 (≤0.7)']
    correlation_counts = [
        validation_results['correlation_distribution']['high'],
        validation_results['correlation_distribution']['medium'],
        validation_results['correlation_distribution']['low']
    ]
    
    correlation_tooltips = [
        f"높은 상관관계 (>0.9)<br>지역 수: {correlation_counts[0]}개<br>BDS가 NAVIS와 매우 유사한 패턴",
        f"중간 상관관계 (0.7-0.9)<br>지역 수: {correlation_counts[1]}개<br>BDS가 NAVIS와 유사하지만 독립적 특성도 보임",
        f"낮은 상관관계 (≤0.7)<br>지역 수: {correlation_counts[2]}개<br>BDS가 NAVIS와 상당히 다른 패턴"
    ]
    
    fig.add_trace(
        go.Bar(
            x=correlation_ranges,
            y=correlation_counts,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            name='상관관계 분포',
            hovertemplate='%{text}<extra></extra>',
            text=correlation_tooltips
        ),
        row=1, col=1
    )
    
    # 1-2. 선행성 우위 지역 (상세 툴팁)
    leading_data = validation_results['leading_top5']
    leading_tooltips = []
    for _, row in leading_data.iterrows():
        tooltip = f"{row['region']}<br>변동성 비율: {row['volatility_ratio']:.3f}<br>상관관계: {row['correlation']:.3f}<br>BDS가 NAVIS보다 {((row['volatility_ratio']-1)*100):.1f}% 더 변동적"
        leading_tooltips.append(tooltip)
    
    fig.add_trace(
        go.Bar(
            x=leading_data['region'],
            y=leading_data['volatility_ratio'],
            marker_color='#96CEB4',
            name='변동성 비율',
            hovertemplate='%{text}<extra></extra>',
            text=leading_tooltips
        ),
        row=1, col=2
    )
    
    # 1-3. Geojson 히트맵
    if geojson:
        # 지역명 매핑
        region_mapping = {
            '서울특별시': 'Seoul',
            '부산광역시': 'Busan',
            '대구광역시': 'Daegu',
            '인천광역시': 'Incheon',
            '광주광역시': 'Gwangju',
            '대전광역시': 'Daejeon',
            '울산광역시': 'Ulsan',
            '세종특별자치시': 'Sejong',
            '경기도': 'Gyeonggi-do',
            '강원도': 'Gangwon-do',
            '충청북도': 'Chungcheongbuk-do',
            '충청남도': 'Chungcheongnam-do',
            '전라북도': 'Jeollabuk-do',
            '전라남도': 'Jeollanam-do',
            '경상북도': 'Gyeongsangbuk-do',
            '경상남도': 'Gyeongsangnam-do',
            '제주특별자치도': 'Jeju-do'
        }
        
        # 데이터 준비
        locations = []
        z_values = []
        hover_texts = []
        
        for _, row in validation_df.iterrows():
            region = row['region']
            if region in region_mapping:
                locations.append(region_mapping[region])
                z_values.append(row['correlation'])
                hover_texts.append(f"{region}<br>상관관계: {row['correlation']:.3f}<br>변동성 비율: {row['volatility_ratio']:.3f}<br>선행성: {'예' if row['is_leading'] else '아니오'}")
        
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=locations,
                z=z_values,
                colorscale='RdYlBu_r',
                featureidkey='properties.CTP_KOR_NM',
                name='상관관계',
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts
            ),
            row=2, col=1
        )
    
    # 1-4. 변동성 비율 산점도 (상세 툴팁)
    volatility_tooltips = []
    for _, row in validation_df.iterrows():
        tooltip = f"{row['region']}<br>상관관계: {row['correlation']:.3f}<br>변동성 비율: {row['volatility_ratio']:.3f}<br>독립성 점수: {row['independence_score']:.3f}<br>선행성: {'예' if row['is_leading'] else '아니오'}"
        volatility_tooltips.append(tooltip)
    
    fig.add_trace(
        go.Scatter(
            x=validation_df['correlation'],
            y=validation_df['volatility_ratio'],
            mode='markers',
            marker=dict(
                size=12,
                color=validation_df['volatility_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="변동성 비율")
            ),
            text=volatility_tooltips,
            hovertemplate='%{text}<extra></extra>',
            name='변동성 vs 상관관계'
        ),
        row=2, col=2
    )
    
    # 1-5. 전 지역 시계열 비교 (1/2)
    regions = bds_df['region'].unique()
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Dark2  # 색상 확장
    
    for i, region in enumerate(regions[:len(regions)//2]):
        region_data = bds_df[bds_df['region'] == region].sort_values('year')
        
        # NAVIS
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['navis_index'],
                mode='lines',
                name=f'NAVIS ({region})',
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # BDS
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['bds_index'],
                mode='lines',
                name=f'BDS ({region})',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 1-6. 전 지역 시계열 비교 (2/2)
    for i, region in enumerate(regions[len(regions)//2:]):
        region_data = bds_df[bds_df['region'] == region].sort_values('year')
        
        # NAVIS
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['navis_index'],
                mode='lines',
                name=f'NAVIS ({region})',
                line=dict(color=colors[(i+len(regions)//2) % len(colors)], width=2),
                showlegend=False
            ),
            row=3, col=2
        )
        
        # BDS
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['bds_index'],
                mode='lines',
                name=f'BDS ({region})',
                line=dict(color=colors[(i+len(regions)//2) % len(colors)], width=2, dash='dash'),
                showlegend=False
            ),
            row=3, col=2
        )
    
    # 1-7. 검증 점수 게이지 (상세 툴팁)
    score_description = f"""
    종합 검증 점수: {validation_results['validation_score']:.3f}
    
    구성 요소:
    • 선행성 점수: {(validation_results['leading_regions'] / validation_results['total_regions']) * 0.4:.3f} (40%)
    • 독립성 점수: {validation_results['avg_independence'] * 0.3:.3f} (30%)
    • 변동성 점수: {(validation_results['avg_volatility_ratio'] - 1) * 0.3:.3f} (30%)
    
    해석:
    • 0.7 이상: 우수한 대체 지표
    • 0.5-0.7: 양호한 보완 지표
    • 0.3-0.5: 개선 필요
    • 0.3 미만: 대체 부적합
    """
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=validation_results['validation_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "종합 검증 점수"},
            delta={'reference': 0.5},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ),
        row=4, col=1
    )
    
    # 1-8. 지역별 성능 등급 (상세 툴팁)
    performance_grades = []
    grade_tooltips = []
    
    for _, row in validation_df.iterrows():
        if row['is_leading'] and row['independence_score'] > 0.1:
            grade = 'A'
            description = "우수: 선행성 + 독립성 모두 우수"
        elif row['is_leading']:
            grade = 'B'
            description = "양호: 선행성 우수, 독립성 보통"
        elif row['independence_score'] > 0.1:
            grade = 'C'
            description = "보통: 독립성 우수, 선행성 부족"
        else:
            grade = 'D'
            description = "개선필요: 선행성과 독립성 모두 부족"
        
        performance_grades.append(grade)
        grade_tooltips.append(f"{row['region']}<br>등급: {grade}<br>{description}")
    
    grade_counts = pd.Series(performance_grades).value_counts()
    grade_descriptions = {
        'A': '우수: 선행성 + 독립성 모두 우수',
        'B': '양호: 선행성 우수, 독립성 보통',
        'C': '보통: 독립성 우수, 선행성 부족',
        'D': '개선필요: 선행성과 독립성 모두 부족'
    }
    
    grade_tooltips = [f"{grade}<br>{grade_descriptions[grade]}<br>지역 수: {count}개" for grade, count in grade_counts.items()]
    
    fig.add_trace(
        go.Bar(
            x=grade_counts.index,
            y=grade_counts.values,
            marker_color=['#28a745', '#17a2b8', '#ffc107', '#dc3545'],
            name='성능 등급',
            hovertemplate='%{text}<extra></extra>',
            text=grade_tooltips
        ),
        row=4, col=2
    )
    
    # 레이아웃 설정 (겹치지 않도록)
    fig.update_layout(
        title={
            'text': '향상된 BDS 모델 종합 분석 대시보드 v2.0',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1800,  # 높이 더 증가
        width=1600,   # 너비 더 증가
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        margin=dict(l=50, r=100, t=120, b=50)  # 여백 더 증가
    )
    
    # 서브플롯 제목 위치 조정
    for i in range(1, 9):
        fig.layout.annotations[i-1].update(
            font=dict(size=14),
            y=fig.layout.annotations[i-1].y + 0.03  # 제목을 더 위로 이동
        )
    
    # 저장
    fig.write_html('enhanced_bds_comprehensive_dashboard_v2.html')
    print("✅ 종합 대시보드 v2.0 저장: enhanced_bds_comprehensive_dashboard_v2.html")
    
    return fig

def create_policy_simulation(bds_df, validation_df):
    """정책 시뮬레이션 기능"""
    print("\n=== 정책 시뮬레이션 생성 ===")
    
    # 1. 투자 효과 시뮬레이션
    def simulate_investment_effect(region_data, investment_amount, investment_type):
        """투자 효과 시뮬레이션"""
        base_bds = region_data['bds_index'].iloc[-1]  # 최신 BDS 값
        
        # 투자 유형별 효과 계수
        effect_coefficients = {
            'infrastructure': 0.15,  # 인프라 투자
            'innovation': 0.20,      # 혁신 투자
            'social': 0.10,          # 사회 투자
            'environmental': 0.08,   # 환경 투자
            'balanced': 0.12         # 균형 투자
        }
        
        effect_coefficient = effect_coefficients[investment_type]
        improvement = investment_amount * effect_coefficient / 1000  # 1000억 단위로 정규화
        
        # 지역별 특성 반영
        if '특별시' in region_data['region'].iloc[0] or '광역시' in region_data['region'].iloc[0]:
            improvement *= 0.8  # 도시는 이미 높은 수준이므로 효과 감소
        elif '도' in region_data['region'].iloc[0]:
            improvement *= 1.2  # 도는 투자 효과가 더 큼
        
        return base_bds + improvement
    
    # 2. 시뮬레이션 시나리오
    scenarios = [
        {'name': '인프라 집중 투자', 'type': 'infrastructure', 'amount': 5000},
        {'name': '혁신 집중 투자', 'type': 'innovation', 'amount': 3000},
        {'name': '사회 복지 투자', 'type': 'social', 'amount': 2000},
        {'name': '환경 친화 투자', 'type': 'environmental', 'amount': 1500},
        {'name': '균형 발전 투자', 'type': 'balanced', 'amount': 4000}
    ]
    
    # 3. 시뮬레이션 결과 생성
    simulation_results = []
    
    for region in bds_df['region'].unique():
        region_data = bds_df[bds_df['region'] == region].sort_values('year')
        current_bds = region_data['bds_index'].iloc[-1]
        
        for scenario in scenarios:
            future_bds = simulate_investment_effect(region_data, scenario['amount'], scenario['type'])
            improvement = ((future_bds - current_bds) / current_bds) * 100
            
            simulation_results.append({
                'region': region,
                'scenario': scenario['name'],
                'investment_type': scenario['type'],
                'investment_amount': scenario['amount'],
                'current_bds': current_bds,
                'future_bds': future_bds,
                'improvement_percent': improvement
            })
    
    simulation_df = pd.DataFrame(simulation_results)
    
    # 4. 시뮬레이션 시각화
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '투자 유형별 평균 개선 효과',
            '지역별 투자 효과 비교 (균형 투자)',
            '투자 금액별 효과 분석',
            '최적 투자 전략 추천'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 4-1. 투자 유형별 평균 개선 효과
    type_effects = simulation_df.groupby('investment_type')['improvement_percent'].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=type_effects['investment_type'],
            y=type_effects['improvement_percent'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            name='평균 개선 효과 (%)',
            hovertemplate='투자 유형: %{x}<br>평균 개선: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 4-2. 지역별 투자 효과 비교 (균형 투자)
    balanced_investment = simulation_df[simulation_df['scenario'] == '균형 발전 투자']
    fig.add_trace(
        go.Bar(
            x=balanced_investment['region'],
            y=balanced_investment['improvement_percent'],
            marker_color='#4ECDC4',
            name='균형 투자 효과 (%)',
            hovertemplate='지역: %{x}<br>개선 효과: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 4-3. 투자 금액별 효과 분석
    amount_effects = simulation_df.groupby('investment_amount')['improvement_percent'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=amount_effects['investment_amount'],
            y=amount_effects['improvement_percent'],
            mode='lines+markers',
            name='투자 금액 vs 효과',
            line=dict(color='#45B7D1', width=3),
            hovertemplate='투자 금액: %{x}억원<br>평균 효과: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4-4. 최적 투자 전략 추천
    # 지역별 최적 투자 유형 찾기
    optimal_strategies = []
    for region in simulation_df['region'].unique():
        region_data = simulation_df[simulation_df['region'] == region]
        best_scenario = region_data.loc[region_data['improvement_percent'].idxmax()]
        optimal_strategies.append({
            'region': region,
            'best_scenario': best_scenario['scenario'],
            'best_improvement': best_scenario['improvement_percent']
        })
    
    optimal_df = pd.DataFrame(optimal_strategies)
    fig.add_trace(
        go.Bar(
            x=optimal_df['region'],
            y=optimal_df['best_improvement'],
            marker_color='#96CEB4',
            name='최적 투자 효과 (%)',
            hovertemplate='지역: %{x}<br>최적 전략: %{text}<br>예상 효과: %{y:.2f}%<extra></extra>',
            text=[f"{row['best_scenario']}" for _, row in optimal_df.iterrows()]
        ),
        row=2, col=2
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': 'BDS 기반 정책 시뮬레이션',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1200,
        width=1400,
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # 저장
    fig.write_html('bds_policy_simulation.html')
    simulation_df.to_csv('bds_policy_simulation_results.csv', index=False, encoding='utf-8-sig')
    
    print("✅ 정책 시뮬레이션 저장:")
    print("  - 시뮬레이션 결과: bds_policy_simulation_results.csv")
    print("  - 시각화: bds_policy_simulation.html")
    
    return simulation_df

def main():
    """메인 실행 함수"""
    print("=== 향상된 BDS 모델 검증 및 시각화 v2.0 ===")
    
    # 1. 데이터 로드
    bds_df, validation_df = load_enhanced_data()
    if bds_df is None or validation_df is None:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. Geojson 로드
    geojson = load_korea_geojson()
    
    # 3. 종합 검증
    validation_results = validate_enhanced_model_comprehensive(bds_df, validation_df)
    
    # 4. 종합 시각화 생성 v2.0
    comprehensive_fig = create_comprehensive_visualization_v2(bds_df, validation_df, validation_results, geojson)
    
    # 5. 정책 시뮬레이션 생성
    simulation_df = create_policy_simulation(bds_df, validation_df)
    
    print(f"\n✅ 향상된 BDS 모델 검증 및 시각화 v2.0 완료!")
    print(f"📊 생성된 파일:")
    print(f"  - 종합 대시보드 v2.0: enhanced_bds_comprehensive_dashboard_v2.html")
    print(f"  - 정책 시뮬레이션: bds_policy_simulation.html")
    print(f"  - 시뮬레이션 결과: bds_policy_simulation_results.csv")
    print(f"\n🏆 주요 성과:")
    print(f"  - 선행성 우위: {validation_results['leading_regions']}개 지역")
    print(f"  - 종합 검증 점수: {validation_results['validation_score']:.3f}")
    print(f"  - 정책 시뮬레이션: {len(simulation_df)}개 시나리오")

if __name__ == "__main__":
    main()
