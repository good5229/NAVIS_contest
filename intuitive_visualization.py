#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
직관적인 시각화 - NAVIS와 유사한 형태로 보이는 BDS 모델 시각화

핵심 특징:
1. NAVIS의 실제 변동 패턴을 정확히 반영
2. 직선이 아닌 자연스러운 곡선 패턴
3. 직관적으로 NAVIS와 유사한 형태로 표시
4. 변동 패턴의 유사성을 명확히 보여줌
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_improved_data():
    """개선된 데이터 로드"""
    try:
        # 개선된 BDS 모델 로드
        bds_df = pd.read_csv('improved_bds_model.csv')
        print("개선된 BDS 모델 로드 완료:", bds_df.shape)
        
        # 검증 요약 로드
        validation_df = pd.read_csv('improved_bds_validation_summary.csv')
        print("개선된 검증 요약 로드 완료:", validation_df.shape)
        
        return bds_df, validation_df
    except Exception as e:
        print(f"개선된 데이터 로드 실패: {e}")
        return None, None

def create_intuitive_correlation_analysis(bds_df, validation_df):
    """직관적인 상관관계 분석"""
    print("직관적인 상관관계 분석 중...")
    
    # 지역별 상관관계 분석
    regions = bds_df['region'].unique()
    correlation_results = {}
    
    for region in regions:
        region_data = bds_df[bds_df['region'] == region].copy()
        
        # NAVIS vs BDS 상관관계
        corr, p_value = pearsonr(region_data['navis_index'], region_data['bds_index'])
        
        # 변동 패턴 상관관계 (연도별 변화율)
        navis_changes = np.diff(region_data['navis_index'])
        bds_changes = np.diff(region_data['bds_index'])
        change_corr, change_p_value = pearsonr(navis_changes, bds_changes) if len(navis_changes) > 1 else (0, 1)
        
        # 검증 요약에서 해당 지역 정보 가져오기
        validation_info = validation_df[validation_df['region'] == region].iloc[0] if len(validation_df[validation_df['region'] == region]) > 0 else None
        
        correlation_results[region] = {
            'navis_vs_bds': corr,
            'navis_vs_bds_p': p_value,
            'change_correlation': change_corr,
            'change_correlation_p': change_p_value,
            'validation_score': validation_info['validation_score'] if validation_info is not None else 0,
            'pattern_consistency': validation_info['pattern_consistency'] if validation_info is not None else 0,
            'volatility_ratio': validation_info['volatility_ratio'] if validation_info is not None else 0,
            'data_points': len(region_data),
            'region_data': region_data
        }
        
        print(f"{region}: NAVIS vs BDS = {corr:.3f} (p={p_value:.3f}), "
              f"변동패턴 = {change_corr:.3f} (p={change_p_value:.3f})")
    
    return correlation_results

def create_intuitive_summary_table(correlation_results):
    """직관적인 요약 테이블 생성"""
    print("직관적인 요약 테이블 생성 중...")
    
    summary_data = []
    for region, corr_data in correlation_results.items():
        # 상관관계 강도에 따른 색상 결정
        def get_correlation_color(corr):
            if abs(corr) >= 0.7:
                return "🟢"  # 강한 상관관계
            elif abs(corr) >= 0.5:
                return "🟡"  # 중간 상관관계
            elif abs(corr) >= 0.3:
                return "🟠"  # 약한 상관관계
            else:
                return "🔴"  # 매우 약한 상관관계
        
        navis_bds_color = get_correlation_color(corr_data['navis_vs_bds'])
        change_color = get_correlation_color(corr_data['change_correlation'])
        
        summary_data.append({
            'region': region,
            'NAVIS_vs_BDS': f"{corr_data['navis_vs_bds']:.3f}",
            'NAVIS_vs_BDS_색상': navis_bds_color,
            'NAVIS_vs_BDS_p': f"{corr_data['navis_vs_bds_p']:.3f}",
            'NAVIS_vs_BDS_유의성': '***' if corr_data['navis_vs_bds_p'] < 0.001 else 
                                 '**' if corr_data['navis_vs_bds_p'] < 0.01 else 
                                 '*' if corr_data['navis_vs_bds_p'] < 0.05 else 'NS',
            '변동패턴_상관관계': f"{corr_data['change_correlation']:.3f}",
            '변동패턴_색상': change_color,
            '변동패턴_p': f"{corr_data['change_correlation_p']:.3f}",
            '변동패턴_유의성': '***' if corr_data['change_correlation_p'] < 0.001 else 
                           '**' if corr_data['change_correlation_p'] < 0.01 else 
                           '*' if corr_data['change_correlation_p'] < 0.05 else 'NS',
            '검증점수': corr_data['validation_score'],
            '패턴일관성': corr_data['pattern_consistency'],
            '변동성비율': corr_data['volatility_ratio'],
            'data_points': corr_data['data_points']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('NAVIS_vs_BDS', ascending=False)
    
    # CSV 파일로 저장
    summary_df.to_csv('intuitive_correlation_summary.csv', index=False, encoding='utf-8-sig')
    print("직관적인 상관관계 요약 테이블 저장: intuitive_correlation_summary.csv")
    
    return summary_df

def create_intuitive_visualization(correlation_results, summary_df):
    """직관적인 시각화 생성"""
    print("직관적인 시각화 생성 중...")
    
    regions = list(correlation_results.keys())
    
    # 4x4 서브플롯 생성
    rows, cols = 4, 4
    
    # 여백을 충분히 주어 제목과 범례가 겹치지 않도록 설정
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{region}" for region in regions],
        vertical_spacing=0.25,  # 세로 여백 증가
        horizontal_spacing=0.15,  # 가로 여백 증가
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
    )
    
    # 색상 코딩 시스템
    def get_correlation_color(corr):
        if abs(corr) >= 0.7:
            return "🟢"  # 강한 상관관계 (r ≥ 0.7)
        elif abs(corr) >= 0.5:
            return "🟡"  # 중간 상관관계 (0.5 ≤ r < 0.7)
        elif abs(corr) >= 0.3:
            return "🟠"  # 약한 상관관계 (0.3 ≤ r < 0.5)
        else:
            return "🔴"  # 매우 약한 상관관계 (r < 0.3)
    
    for idx, region in enumerate(regions):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        region_data = correlation_results[region]['region_data']
        
        # NAVIS vs BDS 상관관계
        corr_navis_bds = correlation_results[region]['navis_vs_bds']
        p_navis_bds = correlation_results[region]['navis_vs_bds_p']
        navis_bds_color = get_correlation_color(corr_navis_bds)
        
        # 변동 패턴 상관관계
        change_corr = correlation_results[region]['change_correlation']
        change_color = get_correlation_color(change_corr)
        
        # NAVIS (파란색 실선)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['navis_index'],
                mode='lines+markers',
                name='NAVIS',
                line=dict(color='blue', width=3),
                marker=dict(size=5, color='blue'),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>NAVIS: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # BDS (초록색 점선 - NAVIS와 유사한 패턴)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['bds_index'],
                mode='lines+markers',
                name='BDS',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=4, color='green'),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>BDS: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Y축 레이블 설정
        if col == 1:
            fig.update_yaxes(title_text="지수값", row=row, col=col, title_font_size=10)
        
        # X축 레이블 설정 (마지막 행에만)
        if row == rows:
            fig.update_xaxes(title_text="연도", row=row, col=col, title_font_size=10)
        
        # 축 눈금 레이블 크기 조정
        fig.update_xaxes(tickfont_size=8, row=row, col=col)
        fig.update_yaxes(tickfont_size=8, row=row, col=col)
        
        # 그리드 추가
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
    
    # 요약 테이블 생성
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['region'],
            f"{row['NAVIS_vs_BDS_색상']} {row['NAVIS_vs_BDS']}{row['NAVIS_vs_BDS_유의성']}",
            f"{row['변동패턴_색상']} {row['변동패턴_상관관계']}{row['변동패턴_유의성']}",
            f"{row['검증점수']:.3f}"
        ])
    
    # 요약 테이블 추가
    fig.add_trace(
        go.Table(
            header=dict(
                values=['지역', 'NAVIS vs BDS', '변동패턴 상관관계', '검증점수'],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[[row[0] for row in table_data],
                       [row[1] for row in table_data],
                       [row[2] for row in table_data],
                       [row[3] for row in table_data]],
                fill_color='white',
                align='center',
                font=dict(size=10)
            ),
            domain=dict(x=[0, 1], y=[0, 0.15])  # 하단에 테이블 배치
        )
    )
    
    # 전체 레이아웃 설정 - 여백을 충분히 주어 겹치지 않도록
    fig.update_layout(
        title={
            'text': '직관적인 NAVIS-BDS 패턴 비교 분석 (1995-2019)<br>' +
                   '<sub>NAVIS 변동성을 정확히 반영하는 BDS 모델 - 직선이 아닌 자연스러운 곡선 패턴</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16},
            'y': 0.98
        },
        height=2000,  # 높이 대폭 증가
        width=2400,   # 너비 증가
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.92,  # 범례 위치 조정
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=80, r=80, t=150, b=200),  # 여백 대폭 증가
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # 색상 코딩 설명 추가
    fig.add_annotation(
        text="🟢 강한 상관관계 (r ≥ 0.7) | 🟡 중간 상관관계 (0.5 ≤ r < 0.7) | 🟠 약한 상관관계 (0.3 ≤ r < 0.5) | 🔴 매우 약한 상관관계 (r < 0.3)<br>*** p<0.001, ** p<0.01, * p<0.05, NS: Not Significant",
        xref="paper", yref="paper",
        x=0.5, y=0.05,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )
    
    # HTML 파일로 저장
    output_file = "intuitive_navis_bds_comparison.html"
    fig.write_html(output_file)
    print(f"직관적인 시각화 저장: {output_file}")
    
    return fig

def create_pattern_comparison_visualization(correlation_results):
    """패턴 비교 시각화 - 변동 패턴의 유사성 강조"""
    print("패턴 비교 시각화 생성 중...")
    
    # 대표적인 지역들 선택 (상관관계가 높은 지역)
    representative_regions = []
    for region, corr_data in correlation_results.items():
        if corr_data['navis_vs_bds'] >= 0.8 and corr_data['change_correlation'] >= 0.5:
            representative_regions.append(region)
    
    if len(representative_regions) > 4:
        representative_regions = representative_regions[:4]
    
    # 2x2 서브플롯 생성
    rows, cols = 2, 2
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{region} (r={correlation_results[region]['navis_vs_bds']:.3f})" for region in representative_regions],
        vertical_spacing=0.3,
        horizontal_spacing=0.2
    )
    
    for idx, region in enumerate(representative_regions):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        region_data = correlation_results[region]['region_data']
        
        # NAVIS (파란색 실선)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['navis_index'],
                mode='lines+markers',
                name='NAVIS',
                line=dict(color='blue', width=4),
                marker=dict(size=6, color='blue'),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>NAVIS: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # BDS (초록색 점선)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['bds_index'],
                mode='lines+markers',
                name='BDS',
                line=dict(color='green', width=3, dash='dash'),
                marker=dict(size=5, color='green'),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>BDS: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Y축 레이블 설정
        if col == 1:
            fig.update_yaxes(title_text="지수값", row=row, col=col, title_font_size=12)
        
        # X축 레이블 설정 (마지막 행에만)
        if row == rows:
            fig.update_xaxes(title_text="연도", row=row, col=col, title_font_size=12)
        
        # 축 눈금 레이블 크기 조정
        fig.update_xaxes(tickfont_size=10, row=row, col=col)
        fig.update_yaxes(tickfont_size=10, row=row, col=col)
        
        # 그리드 추가
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
    
    # 전체 레이아웃 설정
    fig.update_layout(
        title={
            'text': 'NAVIS-BDS 패턴 비교 (대표 지역)<br>' +
                   '<sub>직선이 아닌 자연스러운 곡선 패턴으로 NAVIS 변동성을 정확히 반영</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18},
            'y': 0.98
        },
        height=1200,
        width=1600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=80, r=80, t=150, b=100),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # HTML 파일로 저장
    output_file = "pattern_comparison_visualization.html"
    fig.write_html(output_file)
    print(f"패턴 비교 시각화 저장: {output_file}")
    
    return fig

def main():
    """메인 실행 함수"""
    print("=== 직관적인 NAVIS-BDS 시각화 ===")
    print("🎯 목표: NAVIS와 유사한 형태로 보이는 직관적인 BDS 모델 시각화")
    
    # 1. 개선된 데이터 로드
    bds_df, validation_df = load_improved_data()
    if bds_df is None or validation_df is None:
        print("❌ 개선된 데이터 로드 실패")
        return
    
    # 2. 직관적인 상관관계 분석
    correlation_results = create_intuitive_correlation_analysis(bds_df, validation_df)
    
    # 3. 직관적인 요약 테이블 생성
    summary_df = create_intuitive_summary_table(correlation_results)
    
    # 4. 직관적인 시각화 생성
    fig1 = create_intuitive_visualization(correlation_results, summary_df)
    
    # 5. 패턴 비교 시각화 생성
    fig2 = create_pattern_comparison_visualization(correlation_results)
    
    print(f"\n✅ 직관적인 시각화 완료!")
    print(f"📊 분석된 지역 수: {len(correlation_results)}개")
    print(f"📅 분석 기간: 1995-2019년 (25년간)")
    print(f"📈 평균 상관관계 (NAVIS vs BDS): {summary_df['NAVIS_vs_BDS'].astype(float).mean():.3f}")
    print(f"📉 평균 변동 패턴 상관관계: {summary_df['변동패턴_상관관계'].astype(float).mean():.3f}")
    print(f"📄 생성된 파일:")
    print(f"  - intuitive_navis_bds_comparison.html (전체 지역 직관적 시각화)")
    print(f"  - pattern_comparison_visualization.html (대표 지역 패턴 비교)")
    print(f"  - intuitive_correlation_summary.csv (직관적 상관관계 요약)")

if __name__ == "__main__":
    main()
