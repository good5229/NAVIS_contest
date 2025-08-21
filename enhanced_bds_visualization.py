#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 BDS 모델 검증 및 시각화

목표:
1. 향상된 BDS 모델의 성능 검증
2. NAVIS 대비 우위성 시각화
3. 선행성과 독립성 분석 시각화
4. 겹치지 않는 깔끔한 레이아웃
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        
        return bds_df, validation_df
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None, None

def validate_enhanced_model_comprehensive(bds_df, validation_df):
    """향상된 모델 종합 검증"""
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
    
    # 4. 검증 점수 계산
    validation_score = (
        (leading_regions / total_regions) * 0.4 +  # 선행성 가중치 40%
        (avg_independence) * 0.3 +  # 독립성 가중치 30%
        (avg_volatility_ratio - 1) * 0.3  # 변동성 가중치 30%
    )
    
    print(f"\n🏅 종합 검증 점수: {validation_score:.3f}")
    
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

def create_comprehensive_visualization(bds_df, validation_df, validation_results):
    """종합 시각화 생성 (겹치지 않는 레이아웃)"""
    print("\n=== 종합 시각화 생성 ===")
    
    # 1. 메인 대시보드 (HTML)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'NAVIS vs BDS 상관관계 분포',
            '선행성 우위 지역 분석',
            '지역별 상관관계 히트맵',
            '변동성 비율 분석',
            'NAVIS vs BDS 시계열 비교 (서울)',
            'NAVIS vs BDS 시계열 비교 (부산)',
            '검증 점수 요약',
            '지역별 성능 등급'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "indicator"}, {"type": "bar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1-1. 상관관계 분포
    correlation_ranges = ['높음 (>0.9)', '중간 (0.7-0.9)', '낮음 (≤0.7)']
    correlation_counts = [
        validation_results['correlation_distribution']['high'],
        validation_results['correlation_distribution']['medium'],
        validation_results['correlation_distribution']['low']
    ]
    
    fig.add_trace(
        go.Bar(
            x=correlation_ranges,
            y=correlation_counts,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
            name='상관관계 분포'
        ),
        row=1, col=1
    )
    
    # 1-2. 선행성 우위 지역
    leading_data = validation_results['leading_top5']
    fig.add_trace(
        go.Bar(
            x=leading_data['region'],
            y=leading_data['volatility_ratio'],
            marker_color='#96CEB4',
            name='변동성 비율'
        ),
        row=1, col=2
    )
    
    # 1-3. 상관관계 히트맵
    regions = validation_df['region'].tolist()
    correlations = validation_df['correlation'].tolist()
    
    fig.add_trace(
        go.Heatmap(
            z=[correlations],
            x=regions,
            colorscale='RdYlBu_r',
            name='상관관계'
        ),
        row=2, col=1
    )
    
    # 1-4. 변동성 비율 산점도
    fig.add_trace(
        go.Scatter(
            x=validation_df['correlation'],
            y=validation_df['volatility_ratio'],
            mode='markers',
            marker=dict(
                size=10,
                color=validation_df['volatility_ratio'],
                colorscale='Viridis',
                showscale=True
            ),
            text=validation_df['region'],
            name='변동성 vs 상관관계'
        ),
        row=2, col=2
    )
    
    # 1-5. 서울 시계열 비교
    seoul_data = bds_df[bds_df['region'] == '서울특별시'].sort_values('year')
    fig.add_trace(
        go.Scatter(
            x=seoul_data['year'],
            y=seoul_data['navis_index'],
            mode='lines+markers',
            name='NAVIS (서울)',
            line=dict(color='#FF6B6B', width=3)
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=seoul_data['year'],
            y=seoul_data['bds_index'],
            mode='lines+markers',
            name='BDS (서울)',
            line=dict(color='#4ECDC4', width=3)
        ),
        row=3, col=1
    )
    
    # 1-6. 부산 시계열 비교
    busan_data = bds_df[bds_df['region'] == '부산광역시'].sort_values('year')
    fig.add_trace(
        go.Scatter(
            x=busan_data['year'],
            y=busan_data['navis_index'],
            mode='lines+markers',
            name='NAVIS (부산)',
            line=dict(color='#FF6B6B', width=3),
            showlegend=False
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=busan_data['year'],
            y=busan_data['bds_index'],
            mode='lines+markers',
            name='BDS (부산)',
            line=dict(color='#4ECDC4', width=3),
            showlegend=False
        ),
        row=3, col=2
    )
    
    # 1-7. 검증 점수 게이지
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
    
    # 1-8. 지역별 성능 등급
    performance_grades = []
    for _, row in validation_df.iterrows():
        if row['is_leading'] and row['independence_score'] > 0.1:
            grade = 'A'
        elif row['is_leading']:
            grade = 'B'
        elif row['independence_score'] > 0.1:
            grade = 'C'
        else:
            grade = 'D'
        performance_grades.append(grade)
    
    grade_counts = pd.Series(performance_grades).value_counts()
    fig.add_trace(
        go.Bar(
            x=grade_counts.index,
            y=grade_counts.values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            name='성능 등급'
        ),
        row=4, col=2
    )
    
    # 레이아웃 설정 (겹치지 않도록)
    fig.update_layout(
        title={
            'text': '향상된 BDS 모델 종합 분석 대시보드',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1600,  # 높이 증가
        width=1400,   # 너비 증가
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        margin=dict(l=50, r=50, t=100, b=50)  # 여백 증가
    )
    
    # 서브플롯 제목 위치 조정
    for i in range(1, 9):
        fig.layout.annotations[i-1].update(
            font=dict(size=14),
            y=fig.layout.annotations[i-1].y + 0.02  # 제목을 위로 이동
        )
    
    # Y축 레이블 간격 조정
    fig.update_yaxes(tickmode='linear', dtick=1)
    
    # 저장
    fig.write_html('enhanced_bds_comprehensive_dashboard.html')
    print("✅ 종합 대시보드 저장: enhanced_bds_comprehensive_dashboard.html")
    
    return fig

def create_regional_comparison_plots(bds_df):
    """지역별 비교 플롯 생성"""
    print("\n=== 지역별 비교 플롯 생성 ===")
    
    # 주요 지역 선택
    major_regions = ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '경기도']
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[f'{region} NAVIS vs BDS 비교' for region in major_regions],
        vertical_spacing=0.12,  # 간격 증가
        horizontal_spacing=0.1
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, region in enumerate(major_regions):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        region_data = bds_df[bds_df['region'] == region].sort_values('year')
        
        # NAVIS
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['navis_index'],
                mode='lines+markers',
                name=f'NAVIS ({region})',
                line=dict(color=colors[i], width=3),
                showlegend=(i == 0)  # 첫 번째만 범례 표시
            ),
            row=row, col=col
        )
        
        # BDS
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['bds_index'],
                mode='lines+markers',
                name=f'BDS ({region})',
                line=dict(color=colors[i], width=3, dash='dash'),
                showlegend=(i == 0)  # 첫 번째만 범례 표시
            ),
            row=row, col=col
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': '주요 지역 NAVIS vs BDS 시계열 비교',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1200,
        width=1400,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle'
        ),
        margin=dict(l=50, r=100, t=100, b=50)  # 오른쪽 여백 증가
    )
    
    # 서브플롯 제목 위치 조정
    for i in range(5):
        fig.layout.annotations[i].update(
            font=dict(size=12),
            y=fig.layout.annotations[i].y + 0.03  # 제목을 위로 이동
        )
    
    # 저장
    fig.write_html('enhanced_bds_regional_comparison.html')
    print("✅ 지역별 비교 플롯 저장: enhanced_bds_regional_comparison.html")
    
    return fig

def create_performance_summary_table(validation_df, validation_results):
    """성능 요약 테이블 생성"""
    print("\n=== 성능 요약 테이블 생성 ===")
    
    # 상세 성능 테이블
    performance_table = validation_df.copy()
    performance_table['성능_등급'] = ''
    
    for i, row in performance_table.iterrows():
        if row['is_leading'] and row['independence_score'] > 0.1:
            grade = 'A (우수)'
        elif row['is_leading']:
            grade = 'B (양호)'
        elif row['independence_score'] > 0.1:
            grade = 'C (보통)'
        else:
            grade = 'D (개선필요)'
        performance_table.loc[i, '성능_등급'] = grade
    
    # 컬럼명 한글화
    performance_table = performance_table.rename(columns={
        'region': '지역',
        'correlation': '상관관계',
        'volatility_ratio': '변동성_비율',
        'independence_score': '독립성_점수',
        'is_leading': '선행성_우위',
        'is_independent': '독립성_우위'
    })
    
    # HTML 테이블 생성
    html_table = performance_table.to_html(
        index=False,
        float_format='%.3f',
        classes=['table', 'table-striped', 'table-hover'],
        table_id='performance-table'
    )
    
    # CSS 스타일 추가
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>향상된 BDS 모델 성능 요약</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .grade-a {{ color: #28a745; font-weight: bold; }}
            .grade-b {{ color: #17a2b8; font-weight: bold; }}
            .grade-c {{ color: #ffc107; font-weight: bold; }}
            .grade-d {{ color: #dc3545; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>향상된 BDS 모델 성능 요약</h1>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{validation_results['leading_regions']}</div>
                    <div class="stat-label">선행성 우위 지역</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{validation_results['validation_score']:.3f}</div>
                    <div class="stat-label">종합 검증 점수</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{validation_results['avg_correlation']:.3f}</div>
                    <div class="stat-label">평균 상관관계</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{validation_results['avg_volatility_ratio']:.3f}</div>
                    <div class="stat-label">평균 변동성 비율</div>
                </div>
            </div>
            
            {html_table}
        </div>
    </body>
    </html>
    """
    
    with open('enhanced_bds_performance_summary.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ 성능 요약 테이블 저장: enhanced_bds_performance_summary.html")
    
    return performance_table

def main():
    """메인 실행 함수"""
    print("=== 향상된 BDS 모델 검증 및 시각화 ===")
    
    # 1. 데이터 로드
    bds_df, validation_df = load_enhanced_data()
    if bds_df is None or validation_df is None:
        print("❌ 데이터 로드 실패")
        return
    
    # 2. 종합 검증
    validation_results = validate_enhanced_model_comprehensive(bds_df, validation_df)
    
    # 3. 종합 시각화 생성
    comprehensive_fig = create_comprehensive_visualization(bds_df, validation_df, validation_results)
    
    # 4. 지역별 비교 플롯 생성
    regional_fig = create_regional_comparison_plots(bds_df)
    
    # 5. 성능 요약 테이블 생성
    performance_table = create_performance_summary_table(validation_df, validation_results)
    
    print(f"\n✅ 향상된 BDS 모델 검증 및 시각화 완료!")
    print(f"📊 생성된 파일:")
    print(f"  - 종합 대시보드: enhanced_bds_comprehensive_dashboard.html")
    print(f"  - 지역별 비교: enhanced_bds_regional_comparison.html")
    print(f"  - 성능 요약: enhanced_bds_performance_summary.html")
    print(f"\n🏆 주요 성과:")
    print(f"  - 선행성 우위: {validation_results['leading_regions']}개 지역")
    print(f"  - 종합 검증 점수: {validation_results['validation_score']:.3f}")
    print(f"  - 평균 변동성 비율: {validation_results['avg_volatility_ratio']:.3f}")

if __name__ == "__main__":
    main()
