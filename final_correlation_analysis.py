#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 상관관계 분석 - 검증된 BDS 모델 사용
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_validated_data():
    """검증된 데이터 로드"""
    try:
        # 검증된 BDS 모델 로드
        bds_df = pd.read_csv('validated_bds_model.csv')
        print("검증된 BDS 모델 로드 완료:", bds_df.shape)
        
        # 검증 요약 로드
        validation_df = pd.read_csv('bds_validation_summary.csv')
        print("검증 요약 로드 완료:", validation_df.shape)
        
        return bds_df, validation_df
    except Exception as e:
        print(f"검증된 데이터 로드 실패: {e}")
        return None, None

def create_final_correlation_analysis(bds_df, validation_df):
    """최종 상관관계 분석"""
    print("최종 상관관계 분석 중...")
    
    # 지역별 상관관계 분석
    regions = bds_df['region'].unique()
    correlation_results = {}
    
    for region in regions:
        region_data = bds_df[bds_df['region'] == region].copy()
        
        # NAVIS vs BDS 상관관계
        corr, p_value = pearsonr(region_data['navis_index'], region_data['bds_index'])
        
        # 학술적 효과 vs BDS 상관관계
        corr_academic, p_value_academic = pearsonr(region_data['academic_effect'], region_data['bds_index'])
        
        # NAVIS vs 학술적 효과 상관관계
        corr_navis_academic, p_value_navis_academic = pearsonr(region_data['navis_index'], region_data['academic_effect'])
        
        correlation_results[region] = {
            'navis_vs_bds': corr,
            'navis_vs_bds_p': p_value,
            'academic_vs_bds': corr_academic,
            'academic_vs_bds_p': p_value_academic,
            'navis_vs_academic': corr_navis_academic,
            'navis_vs_academic_p': p_value_navis_academic,
            'data_points': len(region_data),
            'region_data': region_data
        }
        
        print(f"{region}: NAVIS vs BDS = {corr:.3f} (p={p_value:.3f})")
    
    return correlation_results

def create_correlation_summary_table(correlation_results, validation_df):
    """상관관계 요약 테이블 생성"""
    print("상관관계 요약 테이블 생성 중...")
    
    summary_data = []
    for region, corr_data in correlation_results.items():
        # 검증 요약에서 해당 지역 정보 가져오기
        validation_info = validation_df[validation_df['region'] == region].iloc[0] if len(validation_df[validation_df['region'] == region]) > 0 else None
        
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
        academic_bds_color = get_correlation_color(corr_data['academic_vs_bds'])
        navis_academic_color = get_correlation_color(corr_data['navis_vs_academic'])
        
        summary_data.append({
            'region': region,
            'NAVIS_vs_BDS': f"{corr_data['navis_vs_bds']:.3f}",
            'NAVIS_vs_BDS_색상': navis_bds_color,
            'NAVIS_vs_BDS_p': f"{corr_data['navis_vs_bds_p']:.3f}",
            'NAVIS_vs_BDS_유의성': '***' if corr_data['navis_vs_bds_p'] < 0.001 else 
                                 '**' if corr_data['navis_vs_bds_p'] < 0.01 else 
                                 '*' if corr_data['navis_vs_bds_p'] < 0.05 else 'NS',
            '학술효과_vs_BDS': f"{corr_data['academic_vs_bds']:.3f}",
            '학술효과_vs_BDS_색상': academic_bds_color,
            '학술효과_vs_BDS_p': f"{corr_data['academic_vs_bds_p']:.3f}",
            '학술효과_vs_BDS_유의성': '***' if corr_data['academic_vs_bds_p'] < 0.001 else 
                                   '**' if corr_data['academic_vs_bds_p'] < 0.01 else 
                                   '*' if corr_data['academic_vs_bds_p'] < 0.05 else 'NS',
            'NAVIS_vs_학술효과': f"{corr_data['navis_vs_academic']:.3f}",
            'NAVIS_vs_학술효과_색상': navis_academic_color,
            'NAVIS_vs_학술효과_p': f"{corr_data['navis_vs_academic_p']:.3f}",
            'NAVIS_vs_학술효과_유의성': '***' if corr_data['navis_vs_academic_p'] < 0.001 else 
                                     '**' if corr_data['navis_vs_academic_p'] < 0.01 else 
                                     '*' if corr_data['navis_vs_academic_p'] < 0.05 else 'NS',
            '검증점수': validation_info['validation_score'] if validation_info is not None else 0,
            '패턴일관성': validation_info['pattern_consistency'] if validation_info is not None else 0,
            'data_points': corr_data['data_points']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('NAVIS_vs_BDS', ascending=False)
    
    # CSV 파일로 저장
    summary_df.to_csv('final_correlation_summary.csv', index=False, encoding='utf-8-sig')
    print("최종 상관관계 요약 테이블 저장: final_correlation_summary.csv")
    
    return summary_df

def create_final_visualization(correlation_results, summary_df):
    """최종 시각화 생성"""
    print("최종 시각화 생성 중...")
    
    regions = list(correlation_results.keys())
    
    # 4x4 서브플롯 생성
    rows, cols = 4, 4
    
    # 여백을 충분히 주어 제목과 범례가 겹치지 않도록 설정
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{region}" for region in regions],
        vertical_spacing=0.25,  # 세로 여백 증가
        horizontal_spacing=0.15,  # 가로 여백 증가
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
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
        
        # NAVIS (왼쪽 Y축)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['navis_index'],
                mode='lines+markers',
                name='NAVIS',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>NAVIS: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col, secondary_y=False
        )
        
        # BDS (오른쪽 Y축)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['bds_index'],
                mode='lines+markers',
                name='BDS',
                line=dict(color='green', width=1.5, dash='dot'),
                marker=dict(size=3),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>BDS: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col, secondary_y=True
        )
        
        # 학술적 효과 (오른쪽 Y축)
        fig.add_trace(
            go.Scatter(
                x=region_data['year'],
                y=region_data['academic_effect'],
                mode='lines+markers',
                name='학술효과',
                line=dict(color='red', width=1.5, dash='dash'),
                marker=dict(size=3),
                showlegend=(idx == 0),
                hovertemplate='<b>%{x}</b><br>학술효과: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col, secondary_y=True
        )
        
        # Y축 레이블 설정
        if col == 1:
            fig.update_yaxes(title_text="NAVIS", secondary_y=False, row=row, col=col, title_font_size=10)
        if col == cols:
            fig.update_yaxes(title_text="BDS/학술효과", secondary_y=True, row=row, col=col, title_font_size=10)
        
        # X축 레이블 설정 (마지막 행에만)
        if row == rows:
            fig.update_xaxes(title_text="연도", row=row, col=col, title_font_size=10)
        
        # 축 눈금 레이블 크기 조정
        fig.update_xaxes(tickfont_size=8, row=row, col=col)
        fig.update_yaxes(tickfont_size=8, row=row, col=col)
    
    # 요약 테이블 생성
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['region'],
            f"{row['NAVIS_vs_BDS_색상']} {row['NAVIS_vs_BDS']}{row['NAVIS_vs_BDS_유의성']}",
            f"{row['학술효과_vs_BDS_색상']} {row['학술효과_vs_BDS']}{row['학술효과_vs_BDS_유의성']}",
            f"{row['NAVIS_vs_학술효과_색상']} {row['NAVIS_vs_학술효과']}{row['NAVIS_vs_학술효과_유의성']}"
        ])
    
    # 요약 테이블 추가
    fig.add_trace(
        go.Table(
            header=dict(
                values=['지역', 'NAVIS vs BDS', '학술효과 vs BDS', 'NAVIS vs 학술효과'],
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
            'text': '검증된 BDS 모델 기반 최종 상관관계 분석 (1995-2019)<br>' +
                   '<sub>검증 통과 모델: 상관관계 0.7-0.95, 검증점수 0.7 이상</sub>',
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
    output_file = "final_correlation_analysis.html"
    fig.write_html(output_file)
    print(f"최종 상관관계 시각화 저장: {output_file}")
    
    return fig

def main():
    """메인 실행 함수"""
    print("=== 검증된 BDS 모델 기반 최종 상관관계 분석 ===")
    
    # 1. 검증된 데이터 로드
    bds_df, validation_df = load_validated_data()
    if bds_df is None or validation_df is None:
        print("❌ 검증된 데이터 로드 실패")
        return
    
    # 2. 최종 상관관계 분석
    correlation_results = create_final_correlation_analysis(bds_df, validation_df)
    
    # 3. 상관관계 요약 테이블 생성
    summary_df = create_correlation_summary_table(correlation_results, validation_df)
    
    # 4. 최종 시각화 생성
    fig = create_final_visualization(correlation_results, summary_df)
    
    print(f"\n✅ 최종 상관관계 분석 완료!")
    print(f"📊 분석된 지역 수: {len(correlation_results)}개")
    print(f"📅 분석 기간: 1995-2019년 (25년간)")
    print(f"📈 평균 상관관계 (NAVIS vs BDS): {summary_df['NAVIS_vs_BDS'].astype(float).mean():.3f}")
    print(f"📉 평균 검증 점수: {summary_df['검증점수'].mean():.3f}")
    print(f"📄 생성된 파일:")
    print(f"  - final_correlation_analysis.html (최종 시각화)")
    print(f"  - final_correlation_summary.csv (최종 상관관계 요약 테이블)")

if __name__ == "__main__":
    main()
