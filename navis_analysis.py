# -*- coding: utf-8 -*-
"""
NAVIS 지역발전지수 분석 및 BDS와의 비교
- NAVIS Excel 데이터 파싱 및 전처리
- BDS와의 상관관계 분석
- 시각화 및 보고서 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_navis_data(file_path: str = "navis_data/1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx"):
    """
    NAVIS Excel 파일을 로드하고 정리합니다.
    """
    try:
        # Excel 파일의 모든 시트 확인
        excel_file = pd.ExcelFile(file_path)
        print(f"발견된 시트: {excel_file.sheet_names}")
        
        # 시트별 데이터 로드 및 정리
        data = {}
        
        for sheet_name in excel_file.sheet_names:
            print(f"\n처리 중: {sheet_name}")
            
            # 시트 로드
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"원본 데이터 형태: {df.shape}")
            print(f"컬럼: {df.columns.tolist()}")
            
            # 데이터 정리
            cleaned_df = clean_navis_sheet(df, sheet_name)
            if cleaned_df is not None:
                data[sheet_name] = cleaned_df
                print(f"정리된 데이터 형태: {cleaned_df.shape}")
        
        return data
        
    except Exception as e:
        print(f"NAVIS 데이터 로드 실패: {e}")
        return {}

def clean_navis_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    NAVIS 시트 데이터를 정리합니다.
    """
    # 첫 번째 행을 헤더로 사용하지 않고, 첫 번째 컬럼이 지역명임
    # 첫 번째 행은 '전국' 등의 메타데이터이므로 제거
    df = df.iloc[1:].reset_index(drop=True)
    
    # 지역명 컬럼 (첫 번째 컬럼)
    region_col = df.columns[0]
    
    # 연도 컬럼들 찾기 (숫자로 된 컬럼들)
    year_cols = []
    for col in df.columns[1:]:  # 첫 번째 컬럼(지역명) 제외
        if str(col).isdigit():
            year_cols.append(col)
    
    if not year_cols:
        print(f"연도 컬럼을 찾을 수 없습니다: {df.columns.tolist()}")
        return None
    
    # 필요한 컬럼만 선택
    selected_cols = [region_col] + year_cols
    df_clean = df[selected_cols].copy()
    
    # 지역명 정규화
    df_clean[region_col] = df_clean[region_col].astype(str).apply(normalize_region_name)
    
    # 수치형 변환
    for col in year_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # 결측값 제거
    df_clean = df_clean.dropna(subset=[region_col])
    
    return df_clean

def normalize_region_name(region: str) -> str:
    """
    지역명을 표준화합니다.
    """
    region = str(region).strip()
    
    # 매핑 딕셔너리
    mapping = {
        '서울': '서울특별시',
        '부산': '부산광역시',
        '대구': '대구광역시',
        '인천': '인천광역시',
        '광주': '광주광역시',
        '대전': '대전광역시',
        '울산': '울산광역시',
        '세종': '세종특별자치시',
        '경기': '경기도',
        '강원': '강원도',
        '충북': '충청북도',
        '충남': '충청남도',
        '전북': '전라북도',
        '전남': '전라남도',
        '경북': '경상북도',
        '경남': '경상남도',
        '제주': '제주특별자치도'
    }
    
    return mapping.get(region, region)

def create_navis_summary(navis_data: dict) -> pd.DataFrame:
    """
    NAVIS 데이터 요약을 생성합니다.
    """
    summary_data = []
    
    for sheet_name, df in navis_data.items():
        # 연도 컬럼들 (지역명 제외)
        year_cols = [col for col in df.columns if col != df.columns[0]]
        
        for year in year_cols:
            year_data = df[['지역명', year]].copy()
            year_data.columns = ['region_name', 'value']
            year_data['year'] = int(year)
            year_data['index_type'] = sheet_name
            summary_data.append(year_data)
    
    if summary_data:
        summary_df = pd.concat(summary_data, ignore_index=True)
        return summary_df
    else:
        return pd.DataFrame()

def compare_bds_with_navis(bds_data: pd.DataFrame, navis_data: dict, target_year: int = 2019):
    """
    BDS와 NAVIS 지수를 비교합니다.
    """
    # BDS 데이터에서 해당 연도 추출
    bds_year = bds_data[bds_data['period'] == target_year].copy()
    
    if bds_year.empty:
        print(f"BDS 데이터에 {target_year}년 데이터가 없습니다.")
        return None
    
    # NAVIS 데이터에서 해당 연도 추출
    navis_year_data = {}
    for sheet_name, df in navis_data.items():
        if str(target_year) in df.columns:
            year_data = df[['지역발전지수', str(target_year)]].copy()
            year_data.columns = ['region_name', 'navis_value']
            year_data = year_data.dropna()
            navis_year_data[sheet_name] = year_data
    
    # 비교 분석
    comparison_results = {}
    
    for navis_type, navis_df in navis_year_data.items():
        # 지역명 매칭
        bds_temp = bds_year.copy()
        bds_temp['region_name'] = bds_temp['region_name'].apply(normalize_region_name)
        navis_df['region_name'] = navis_df['region_name'].apply(normalize_region_name)
        
        # 데이터 병합
        merged = bds_temp.merge(navis_df, on='region_name', how='inner')
        
        if len(merged) >= 3:
            # 상관관계 분석
            correlation = merged['BDS'].corr(merged['navis_value'])
            rank_correlation = merged['BDS'].corr(merged['navis_value'], method='spearman')
            
            # 순위 일치도
            bds_rank = merged['BDS'].rank(ascending=False)
            navis_rank = merged['navis_value'].rank(ascending=False)
            rank_agreement = 1 - (np.abs(bds_rank - navis_rank) / len(merged)).mean()
            
            comparison_results[navis_type] = {
                'correlation': correlation,
                'rank_correlation': rank_correlation,
                'rank_agreement': rank_agreement,
                'common_regions': len(merged),
                'regions': merged['region_name'].tolist(),
                'merged_data': merged
            }
        else:
            comparison_results[navis_type] = {
                'error': f'공통 지역이 부족합니다. (공통 지역: {len(merged)})'
            }
    
    return comparison_results

def visualize_comparison(comparison_results: dict, target_year: int = 2019):
    """
    BDS와 NAVIS 지수의 비교를 시각화합니다.
    """
    # 유효한 결과만 필터링
    valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
    
    if not valid_results:
        print("시각화할 유효한 비교 결과가 없습니다.")
        return None
    
    # 서브플롯 생성
    n_plots = len(valid_results)
    fig = make_subplots(
        rows=2, cols=n_plots,
        subplot_titles=[f'BDS vs {key}' for key in valid_results.keys()] + 
                      [f'{key} 상관관계' for key in valid_results.keys()],
        specs=[[{"secondary_y": False}] * n_plots,
               [{"secondary_y": False}] * n_plots],
        vertical_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (navis_type, result) in enumerate(valid_results.items(), 1):
        merged_data = result['merged_data']
        
        # 산점도 (첫 번째 행)
        fig.add_trace(
            go.Scatter(
                x=merged_data['BDS'], 
                y=merged_data['navis_value'],
                mode='markers+text',
                text=merged_data['region_name'],
                textposition="top center",
                name=navis_type,
                marker=dict(color=colors[i-1], size=12, opacity=0.7),
                showlegend=False,
                hovertemplate='<b>%{text}</b><br>' +
                            'BDS: %{x:.3f}<br>' +
                            f'{navis_type}: %{{y:.3f}}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=i
        )
        
        # 상관계수 표시
        corr = result['correlation']
        corr_interpretation = get_correlation_interpretation(corr)
        
        fig.add_annotation(
            x=0.5, y=0.95,
            xref=f'x{i}', yref=f'y{i}',
            text=f'피어슨 상관계수: {corr:.3f}<br>순위상관: {result["rank_correlation"]:.3f}<br><b>{corr_interpretation}</b>',
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        )
        
        # 상관관계 막대차트 (두 번째 행)
        fig.add_trace(
            go.Bar(
                x=['피어슨', '스피어만', '순위일치'],
                y=[result['correlation'], result['rank_correlation'], result['rank_agreement']],
                name=navis_type,
                marker_color=colors[i-1],
                showlegend=False,
                hovertemplate='<b>%{x}</b><br>' +
                            '상관계수: %{y:.3f}<br>' +
                            '<extra></extra>'
            ),
            row=2, col=i
        )
        
        # 축 레이블
        fig.update_xaxes(title_text="BDS", row=1, col=i)
        fig.update_yaxes(title_text=f"{navis_type}", row=1, col=i)
        fig.update_xaxes(title_text="상관관계 지표", row=2, col=i)
        fig.update_yaxes(title_text="상관계수", row=2, col=i, range=[0, 1])
    
    fig.update_layout(
        title=f'BDS와 NAVIS 지역발전지수 비교 분석 ({target_year}년)<br><sub>피어슨 상관계수: 선형 관계의 강도 | 스피어만 순위상관: 순위 관계의 강도 | 순위일치: 순위 일치도</sub>',
        height=800,
        showlegend=False
    )
    
    return fig

def get_correlation_interpretation(correlation):
    """상관계수 해석을 반환합니다."""
    abs_corr = abs(correlation)
    if abs_corr >= 0.8:
        strength = "매우 강한"
    elif abs_corr >= 0.6:
        strength = "강한"
    elif abs_corr >= 0.4:
        strength = "중간"
    elif abs_corr >= 0.2:
        strength = "약한"
    else:
        strength = "매우 약한"
    
    direction = "양의" if correlation > 0 else "음의"
    return f"{strength} {direction} 상관관계"

def visualize_simulation(simulation_data: pd.DataFrame, target_regions: list):
    """
    시뮬레이션 결과를 시각화합니다.
    """
    # 기본 라인 플롯 (모든 지역을 동일한 색상으로)
    fig = px.line(simulation_data, x='period', y='BDS', color='region_name',
                  title='지역 균형 개선 시뮬레이션',
                  labels={'period': '연도', 'BDS': 'BDS 점수', 'region_name': '지역'},
                  color_discrete_map={region: '#1f77b4' for region in simulation_data['region_name'].unique()})
    
    # 대상 지역 강조 (점선과 마커로 구분)
    for region in target_regions:
        region_data = simulation_data[simulation_data['region_name'] == region]
        fig.add_trace(go.Scatter(
            x=region_data['period'], 
            y=region_data['BDS'],
            mode='lines+markers', 
            name=f'{region} (개선대상)',
            line=dict(width=4, dash='dash', color='red'), 
            marker=dict(size=10, color='red'),
            showlegend=True,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        '연도: %{x}<br>' +
                        'BDS: %{y:.3f}<br>' +
                        '<extra></extra>'
        ))
    
    # 기존 라인들의 범례 숨기기 (대상 지역만 표시)
    for trace in fig.data:
        if trace.name not in [f'{region} (개선대상)' for region in target_regions]:
            trace.showlegend = False
            trace.hovertemplate = '<b>%{fullData.name}</b><br>' + \
                                '연도: %{x}<br>' + \
                                'BDS: %{y:.3f}<br>' + \
                                '<extra></extra>'
    
    fig.update_layout(
        height=600,
        title={
            'text': '지역 균형 개선 시뮬레이션<br><sub>빨간 점선: 개선 대상 지역 (연 10% BDS 향상) | 절차: 1) 하위 5개 지역 선정 2) 연간 10% 개선 3) 5년간 시뮬레이션</sub>',
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig

def create_simulation_dashboard(simulation_data: pd.DataFrame, target_regions: list, balance_analysis: dict):
    """
    시뮬레이션 대시보드를 생성합니다.
    """
    # 1. 시뮬레이션 라인 차트
    fig1 = px.line(simulation_data, x='period', y='BDS', color='region_name',
                   title='지역별 BDS 변화 추이',
                   labels={'period': '연도', 'BDS': 'BDS 점수', 'region_name': '지역'},
                   color_discrete_map={region: '#1f77b4' for region in simulation_data['region_name'].unique()})
    
    # 대상 지역 강조
    for region in target_regions:
        region_data = simulation_data[simulation_data['region_name'] == region]
        fig1.add_trace(go.Scatter(
            x=region_data['period'], 
            y=region_data['BDS'],
            mode='lines+markers', 
            name=f'{region} (개선대상)',
            line=dict(width=4, dash='dash', color='red'), 
            marker=dict(size=10, color='red'),
            showlegend=True,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        '연도: %{x}<br>' +
                        'BDS: %{y:.3f}<br>' +
                        '<extra></extra>'
        ))
    
    # 기존 라인들의 범례 숨기기
    for trace in fig1.data:
        if trace.name not in [f'{region} (개선대상)' for region in target_regions]:
            trace.showlegend = False
            trace.hovertemplate = '<b>%{fullData.name}</b><br>' + \
                                '연도: %{x}<br>' + \
                                'BDS: %{y:.3f}<br>' + \
                                '<extra></extra>'
    
    fig1.update_layout(
        title='지역별 BDS 변화 추이<br><sub>절차: 1) 2013-2022년 실제 데이터 2) 2023-2027년 시뮬레이션 (하위 지역 10% 개선)</sub>'
    )
    
    # 2. 최종 연도 비교 차트
    final_year = simulation_data['period'].max()
    final_data = simulation_data[simulation_data['period'] == final_year].copy()
    final_data['is_target'] = final_data['region_name'].isin(target_regions)
    
    fig2 = px.bar(final_data, x='region_name', y='BDS', color='is_target',
                  title=f'{final_year}년 지역별 BDS 점수',
                  labels={'region_name': '지역', 'BDS': 'BDS 점수', 'is_target': '개선대상'},
                  color_discrete_map={True: 'red', False: '#1f77b4'})
    
    # hovertemplate 추가
    fig2.update_traces(
        hovertemplate='<b>%{x}</b><br>BDS: %{y:.3f}<br><extra></extra>'
    )
    
    fig2.update_layout(
        title=f'{final_year}년 지역별 BDS 점수<br><sub>절차: 시뮬레이션 최종 연도 결과 | 빨간색: 개선 대상 지역</sub>'
    )
    
    # 3. 개선 효과 요약 차트
    improvement_data = []
    for region in target_regions:
        region_data = simulation_data[simulation_data['region_name'] == region]
        initial_bds = region_data[region_data['period'] == 2022]['BDS'].iloc[0]
        final_bds = region_data[region_data['period'] == final_year]['BDS'].iloc[0]
        improvement = ((final_bds - initial_bds) / abs(initial_bds)) * 100
        
        improvement_data.append({
            'region': region,
            'initial_bds': initial_bds,
            'final_bds': final_bds,
            'improvement_pct': improvement
        })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    fig3 = px.bar(improvement_df, x='region', y='improvement_pct',
                  title='개선 대상 지역별 BDS 향상률',
                  labels={'region': '지역', 'improvement_pct': '향상률 (%)'},
                  color='improvement_pct',
                  color_continuous_scale='Reds')
    
    # hovertemplate 추가
    fig3.update_traces(
        hovertemplate='<b>%{x}</b><br>향상률: %{y:.1f}%<br><extra></extra>'
    )
    
    fig3.update_layout(
        title='개선 대상 지역별 BDS 향상률<br><sub>절차: (최종 BDS - 초기 BDS) / |초기 BDS| × 100</sub>'
    )
    
    # 4. 균형성 지표 변화 차트
    years = sorted(simulation_data['period'].unique())
    balance_metrics = []
    
    for year in years:
        year_data = simulation_data[simulation_data['period'] == year]
        bds_values = year_data['BDS'].values
        
        # BDS 값들이 이미 표준화되어 있으므로 절댓값을 사용
        mean_abs = np.mean(np.abs(bds_values))
        std_val = np.std(bds_values)
        
        metrics = {
            'year': year,
            'mean': np.mean(bds_values),
            'std': std_val,
            'cv': std_val / mean_abs if mean_abs > 0 else 0,
            'gini': calculate_gini_coefficient(bds_values),
            'max_min_ratio': np.max(bds_values) / np.min(bds_values) if np.min(bds_values) != 0 else 0
        }
        balance_metrics.append(metrics)
    
    balance_df = pd.DataFrame(balance_metrics)
    
    fig4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=['평균 BDS', '표준편차', '변동계수', '지니계수'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 평균 BDS
    fig4.add_trace(go.Scatter(
        x=balance_df['year'], y=balance_df['mean'], name='평균',
        hovertemplate='연도: %{x}<br>평균 BDS: %{y:.3f}<br><extra></extra>'
    ), row=1, col=1)
    
    # 표준편차
    fig4.add_trace(go.Scatter(
        x=balance_df['year'], y=balance_df['std'], name='표준편차',
        hovertemplate='연도: %{x}<br>표준편차: %{y:.3f}<br><extra></extra>'
    ), row=1, col=2)
    
    # 변동계수
    fig4.add_trace(go.Scatter(
        x=balance_df['year'], y=balance_df['cv'], name='변동계수',
        hovertemplate='연도: %{x}<br>변동계수: %{y:.3f}<br>정의: 표준편차/절댓값평균<br><extra></extra>'
    ), row=2, col=1)
    
    # 지니계수
    fig4.add_trace(go.Scatter(
        x=balance_df['year'], y=balance_df['gini'], name='지니계수',
        hovertemplate='연도: %{x}<br>지니계수: %{y:.3f}<br>정의: 불평등도 (0=평등, 1=불평등)<br><extra></extra>'
    ), row=2, col=2)
    
    fig4.update_layout(
        title='균형성 지표 변화 추이<br><sub>절차: 연도별 BDS 데이터로 지표 계산 | 변동계수: 낮을수록 균형 | 지니계수: 낮을수록 평등</sub>',
        height=600,
        showlegend=False
    )
    
    return fig1, fig2, fig3, fig4

def create_summary_visualization(bds_data: pd.DataFrame, target_regions: list):
    """
    요약 시각화를 생성합니다.
    """
    # 최신 연도 데이터
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    latest_data['is_target'] = latest_data['region_name'].isin(target_regions)
    
    # 1. 현재 상황 요약
    fig1 = px.bar(latest_data, x='region_name', y='BDS', color='is_target',
                  title=f'{latest_year}년 현재 지역별 BDS 점수',
                  labels={'region_name': '지역', 'BDS': 'BDS 점수', 'is_target': '개선대상'},
                  color_discrete_map={True: 'red', False: '#1f77b4'})
    
    # hovertemplate 추가
    fig1.update_traces(
        hovertemplate='<b>%{x}</b><br>BDS: %{y:.3f}<br><extra></extra>'
    )
    
    fig1.update_layout(
        title=f'{latest_year}년 현재 지역별 BDS 점수<br><sub>절차: 최신 연도 데이터 추출 | 빨간색: 개선 대상 지역 (하위 5개)</sub>'
    )
    
    # 2. 지역별 4개 지표 비교
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=['1인당 지역내총생산', '인구증가율', '고령인구비율', '재정자립도'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 대상 지역 강조
    target_data = latest_data[latest_data['is_target']]
    other_data = latest_data[~latest_data['is_target']]
    
    # 1인당 지역내총생산
    fig2.add_trace(go.Bar(
        x=other_data['region_name'], y=other_data['pc_grdp'], 
        name='기타 지역', marker_color='#1f77b4',
        hovertemplate='<b>%{x}</b><br>1인당 지역내총생산: %{y:,.0f}원<br><extra></extra>'
    ), row=1, col=1)
    fig2.add_trace(go.Bar(
        x=target_data['region_name'], y=target_data['pc_grdp'], 
        name='개선대상', marker_color='red',
        hovertemplate='<b>%{x}</b><br>1인당 지역내총생산: %{y:,.0f}원<br><extra></extra>'
    ), row=1, col=1)
    
    # 인구증가율
    fig2.add_trace(go.Bar(
        x=other_data['region_name'], y=other_data['pop_growth'], 
        name='기타 지역', marker_color='#1f77b4', showlegend=False,
        hovertemplate='<b>%{x}</b><br>인구증가율: %{y:.2f}%<br><extra></extra>'
    ), row=1, col=2)
    fig2.add_trace(go.Bar(
        x=target_data['region_name'], y=target_data['pop_growth'], 
        name='개선대상', marker_color='red', showlegend=False,
        hovertemplate='<b>%{x}</b><br>인구증가율: %{y:.2f}%<br><extra></extra>'
    ), row=1, col=2)
    
    # 고령인구비율
    fig2.add_trace(go.Bar(
        x=other_data['region_name'], y=other_data['elderly_rate'], 
        name='기타 지역', marker_color='#1f77b4', showlegend=False,
        hovertemplate='<b>%{x}</b><br>고령인구비율: %{y:.1f}%<br><extra></extra>'
    ), row=2, col=1)
    fig2.add_trace(go.Bar(
        x=target_data['region_name'], y=target_data['elderly_rate'], 
        name='개선대상', marker_color='red', showlegend=False,
        hovertemplate='<b>%{x}</b><br>고령인구비율: %{y:.1f}%<br><extra></extra>'
    ), row=2, col=1)
    
    # 재정자립도
    fig2.add_trace(go.Bar(
        x=other_data['region_name'], y=other_data['fiscal_indep'], 
        name='기타 지역', marker_color='#1f77b4', showlegend=False,
        hovertemplate='<b>%{x}</b><br>재정자립도: %{y:.1f}%<br><extra></extra>'
    ), row=2, col=2)
    fig2.add_trace(go.Bar(
        x=target_data['region_name'], y=target_data['fiscal_indep'], 
        name='개선대상', marker_color='red', showlegend=False,
        hovertemplate='<b>%{x}</b><br>재정자립도: %{y:.1f}%<br><extra></extra>'
    ), row=2, col=2)
    
    fig2.update_layout(
        title='지역별 4개 핵심 지표 비교<br><sub>절차: 최신 연도 데이터로 4개 지표 비교 | 빨간색: 개선 대상 지역</sub>',
        height=800
    )
    
    return fig1, fig2

def create_regional_balance_simulation(bds_data: pd.DataFrame, 
                                     target_regions: list = None,
                                     improvement_rate: float = 0.05,  # 10% → 5%로 조정
                                     simulation_years: int = 5):
    """
    지역 균형 개선 시뮬레이션을 생성합니다. (개선된 버전)
    """
    # 최신 연도 데이터
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    
    # 개선 대상 지역 선정
    if target_regions is None:
        target_regions = latest_data.nsmallest(5, 'BDS')['region_name'].tolist()
    
    # 시뮬레이션 데이터 생성
    simulation_data = []
    
    for year in range(latest_year + 1, latest_year + simulation_years + 1):
        year_data = latest_data.copy()
        year_data['period'] = year
        
        # 대상 지역들의 BDS 개선 (더 현실적인 모델)
        for region in target_regions:
            mask = year_data['region_name'] == region
            if mask.any():
                years_passed = year - latest_year
                
                # 개선 잠재력 기반 차별화된 개선률 적용
                region_data = year_data[mask].iloc[0]
                improvement_potential = calculate_single_improvement_potential(region_data)
                
                # 개선 잠재력이 높을수록 더 큰 개선 효과
                adjusted_improvement_rate = improvement_rate * (1 + improvement_potential)
                
                # 체감 효과 (시간이 지날수록 개선 효과 감소)
                diminishing_factor = 1 / (1 + years_passed * 0.1)
                final_improvement_rate = adjusted_improvement_rate * diminishing_factor
                
                # BDS 개선 적용
                improvement_factor = 1 + final_improvement_rate
                year_data.loc[mask, 'BDS'] *= improvement_factor
        
        simulation_data.append(year_data)
    
    # 결과 결합
    result = pd.concat([bds_data, pd.concat(simulation_data, ignore_index=True)], 
                      ignore_index=True)
    
    return result, target_regions

def create_multiple_simulation_scenarios(bds_data: pd.DataFrame, target_regions: list):
    """
    다양한 시나리오의 시뮬레이션을 생성합니다.
    """
    scenarios = {
        'conservative': {'improvement_rate': 0.03, 'name': '보수적 시나리오 (3%)'},
        'moderate': {'improvement_rate': 0.05, 'name': '중간 시나리오 (5%)'},
        'aggressive': {'improvement_rate': 0.08, 'name': '적극적 시나리오 (8%)'},
        'targeted': {'improvement_rate': 0.06, 'name': '차별화 시나리오 (6%)'}
    }
    
    simulation_results = {}
    
    for scenario_name, scenario_config in scenarios.items():
        if scenario_name == 'targeted':
            # 차별화 시나리오: 개선 잠재력에 따른 차별화된 투자
            simulation_data, _ = create_targeted_improvement_simulation(bds_data, target_regions)
        else:
            simulation_data, _ = create_regional_balance_simulation(
                bds_data, target_regions, scenario_config['improvement_rate']
            )
        
        simulation_results[scenario_name] = {
            'data': simulation_data,
            'name': scenario_config['name'],
            'improvement_rate': scenario_config['improvement_rate']
        }
    
    return simulation_results

def create_targeted_improvement_simulation(bds_data: pd.DataFrame, target_regions: list):
    """
    개선 잠재력에 따른 차별화된 개선 시뮬레이션
    """
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    
    # 각 지역의 개선 잠재력 계산
    region_potentials = {}
    for region in target_regions:
        region_data = latest_data[latest_data['region_name'] == region]
        if not region_data.empty:
            potential = calculate_single_improvement_potential(region_data.iloc[0])
            region_potentials[region] = potential
    
    # 개선 잠재력에 따른 개선률 차별화
    max_potential = max(region_potentials.values()) if region_potentials else 1
    min_potential = min(region_potentials.values()) if region_potentials else 0
    
    simulation_data = []
    
    for year in range(latest_year + 1, latest_year + 6):
        year_data = latest_data.copy()
        year_data['period'] = year
        
        for region in target_regions:
            mask = year_data['region_name'] == region
            if mask.any():
                years_passed = year - latest_year
                potential = region_potentials.get(region, 0)
                
                # 개선 잠재력에 따른 차별화된 개선률
                if max_potential > min_potential:
                    normalized_potential = (potential - min_potential) / (max_potential - min_potential)
                    improvement_rate = 0.03 + (normalized_potential * 0.06)  # 3%~9% 범위
                else:
                    improvement_rate = 0.06  # 기본값
                
                # 체감 효과 적용
                diminishing_factor = 1 / (1 + years_passed * 0.15)
                final_improvement_rate = improvement_rate * diminishing_factor
                
                improvement_factor = 1 + final_improvement_rate
                year_data.loc[mask, 'BDS'] *= improvement_factor
        
        simulation_data.append(year_data)
    
    result = pd.concat([bds_data, pd.concat(simulation_data, ignore_index=True)], 
                      ignore_index=True)
    
    return result, target_regions

def analyze_regional_balance_indicators(bds_data: pd.DataFrame):
    """
    지역 균형성 지표들을 분석합니다.
    """
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year]
    
    bds_values = latest_data['BDS'].values
    
    # BDS 값들이 이미 표준화되어 있으므로 절댓값을 사용하여 변동계수 계산
    mean_abs = np.mean(np.abs(bds_values))
    std_val = np.std(bds_values)
    
    # 다양한 균형성 지표 계산
    analysis = {
        'year': latest_year,
        'mean': np.mean(bds_values),
        'std': std_val,
        'cv': std_val / mean_abs if mean_abs > 0 else 0,  # 절댓값 평균으로 변동계수 계산
        'gini': calculate_gini_coefficient(bds_values),
        'max_min_ratio': np.max(bds_values) / np.min(bds_values) if np.min(bds_values) != 0 else 0,
        'top_bottom_ratio': np.percentile(bds_values, 80) / np.percentile(bds_values, 20) if np.percentile(bds_values, 20) != 0 else 0,
        'iqr': np.percentile(bds_values, 75) - np.percentile(bds_values, 25),
        'skewness': stats.skew(bds_values),
        'kurtosis': stats.kurtosis(bds_values),
        'range': np.max(bds_values) - np.min(bds_values),
        'mean_abs': mean_abs
    }
    
    return analysis

def calculate_gini_coefficient(values: np.ndarray) -> float:
    """지니계수 계산 - 수정된 버전"""
    if len(values) == 0:
        return 0.0
    
    # 음수 값들을 양수로 변환하여 계산 (절댓값 사용)
    abs_values = np.abs(values)
    sorted_values = np.sort(abs_values)
    n = len(sorted_values)
    
    if n == 0 or np.sum(sorted_values) == 0:
        return 0.0
    
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def generate_analysis_report(bds_data: pd.DataFrame, navis_data: dict, 
                           comparison_results: dict, balance_analysis: dict):
    """
    종합 분석 보고서를 생성합니다.
    """
    report = []
    report.append("# BDS와 NAVIS 지역발전지수 비교 분석 보고서\n")
    
    # 1. 지역 균형성 분석
    report.append("## 1. 지역 균형성 분석\n")
    report.append(f"- 분석 연도: {balance_analysis['year']}")
    report.append(f"- 평균 BDS: {balance_analysis['mean']:.4f}")
    report.append(f"- 표준편차: {balance_analysis['std']:.4f}")
    report.append(f"- 변동계수: {balance_analysis['cv']:.4f}")
    report.append(f"- 지니계수: {balance_analysis['gini']:.4f}")
    report.append(f"- 최대/최소 비율: {balance_analysis['max_min_ratio']:.4f}")
    report.append(f"- 상위20%/하위20% 비율: {balance_analysis['top_bottom_ratio']:.4f}\n")
    
    # 2. NAVIS 지수와의 비교
    report.append("## 2. NAVIS 지역발전지수와의 비교\n")
    
    for navis_type, result in comparison_results.items():
        report.append(f"### {navis_type}\n")
        
        if 'error' in result:
            report.append(f"- 오류: {result['error']}\n")
        else:
            report.append(f"- 피어슨 상관계수: {result['correlation']:.4f}")
            report.append(f"- 스피어만 순위상관계수: {result['rank_correlation']:.4f}")
            report.append(f"- 순위 일치도: {result['rank_agreement']:.4f}")
            report.append(f"- 공통 지역 수: {result['common_regions']}")
            report.append(f"- 공통 지역: {', '.join(result['regions'])}\n")
    
    # 3. 해석 및 제언
    report.append("## 3. 해석 및 제언\n")
    
    # 상관관계 해석
    valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
    if valid_results:
        avg_correlation = np.mean([r['correlation'] for r in valid_results.values()])
        report.append(f"- 평균 상관계수: {avg_correlation:.4f}")
        
        if avg_correlation > 0.7:
            report.append("- BDS와 NAVIS 지수는 높은 상관관계를 보입니다.")
        elif avg_correlation > 0.5:
            report.append("- BDS와 NAVIS 지수는 중간 정도의 상관관계를 보입니다.")
        else:
            report.append("- BDS와 NAVIS 지수는 낮은 상관관계를 보입니다.")
    
    # 균형성 해석
    if balance_analysis['cv'] > 0.5:
        report.append("- 지역 간 BDS 차이가 상당히 큽니다.")
    elif balance_analysis['cv'] > 0.3:
        report.append("- 지역 간 BDS 차이가 중간 정도입니다.")
    else:
        report.append("- 지역 간 BDS 차이가 상대적으로 작습니다.")
    
    return "\n".join(report)

def analyze_investment_priority(bds_data: pd.DataFrame, target_regions: list):
    """
    투자 우선순위를 분석합니다.
    """
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    
    # 개선 대상 지역들의 현재 상황 분석
    target_analysis = []
    
    for region in target_regions:
        region_data = latest_data[latest_data['region_name'] == region]
        if not region_data.empty:
            bds_score = region_data['BDS'].iloc[0]
            pc_grdp = region_data['pc_grdp'].iloc[0]
            pop_growth = region_data['pop_growth'].iloc[0]
            elderly_rate = region_data['elderly_rate'].iloc[0]
            fiscal_indep = region_data['fiscal_indep'].iloc[0]
            
            # 투자 우선순위 점수 계산 (BDS 점수가 낮을수록 높은 우선순위)
            priority_score = abs(bds_score)  # 절댓값 사용
            
            # 개선 잠재력 계산 (각 지표별 개선 여지)
            improvement_potential = {
                'pc_grdp': max(0, (50000 - pc_grdp) / 50000),  # 5만원 기준
                'pop_growth': max(0, (1.0 - pop_growth) / 1.0),  # 1% 기준
                'elderly_rate': max(0, (elderly_rate - 10) / 20),  # 10% 기준
                'fiscal_indep': max(0, (50 - fiscal_indep) / 50)  # 50% 기준
            }
            
            total_potential = sum(improvement_potential.values()) / len(improvement_potential)
            
            target_analysis.append({
                'region': region,
                'bds_score': bds_score,
                'priority_score': priority_score,
                'improvement_potential': total_potential,
                'pc_grdp': pc_grdp,
                'pop_growth': pop_growth,
                'elderly_rate': elderly_rate,
                'fiscal_indep': fiscal_indep,
                'overall_priority': priority_score * total_potential
            })
    
    # 우선순위 순으로 정렬
    target_analysis.sort(key=lambda x: x['overall_priority'], reverse=True)
    
    return target_analysis

def create_investment_priority_visualization(priority_analysis: list):
    """
    투자 우선순위 시각화를 생성합니다.
    """
    if not priority_analysis:
        return None
    
    df = pd.DataFrame(priority_analysis)
    
    # 1. 투자 우선순위 차트
    fig1 = px.bar(df, x='region', y='overall_priority',
                  title='투자 우선순위 분석',
                  labels={'region': '지역', 'overall_priority': '투자 우선순위 점수'},
                  color='overall_priority',
                  color_continuous_scale='Reds')
    
    fig1.update_traces(
        hovertemplate='<b>%{x}</b><br>우선순위 점수: %{y:.3f}<br><extra></extra>'
    )
    
    fig1.update_layout(
        title='투자 우선순위 분석<br><sub>절차: BDS 점수 × 개선 잠재력 | 높을수록 투자 우선순위 높음</sub>'
    )
    
    # 2. 개선 잠재력 분석
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=['1인당 지역내총생산', '인구증가율', '고령인구비율', '재정자립도'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1인당 지역내총생산
    fig2.add_trace(go.Bar(
        x=df['region'], y=df['pc_grdp'],
        name='1인당 지역내총생산',
        hovertemplate='<b>%{x}</b><br>1인당 지역내총생산: %{y:,.0f}원<br><extra></extra>'
    ), row=1, col=1)
    
    # 인구증가율
    fig2.add_trace(go.Bar(
        x=df['region'], y=df['pop_growth'],
        name='인구증가율',
        hovertemplate='<b>%{x}</b><br>인구증가율: %{y:.2f}%<br><extra></extra>'
    ), row=1, col=2)
    
    # 고령인구비율
    fig2.add_trace(go.Bar(
        x=df['region'], y=df['elderly_rate'],
        name='고령인구비율',
        hovertemplate='<b>%{x}</b><br>고령인구비율: %{y:.1f}%<br><extra></extra>'
    ), row=2, col=1)
    
    # 재정자립도
    fig2.add_trace(go.Bar(
        x=df['region'], y=df['fiscal_indep'],
        name='재정자립도',
        hovertemplate='<b>%{x}</b><br>재정자립도: %{y:.1f}%<br><extra></extra>'
    ), row=2, col=2)
    
    fig2.update_layout(
        title='개선 대상 지역별 4개 지표 현황<br><sub>절차: 각 지표별 현재 수준 분석 | 개선 여지가 큰 지표 파악</sub>',
        height=600,
        showlegend=False
    )
    
    return fig1, fig2

def validate_improvement_potential(bds_data: pd.DataFrame):
    """
    개선 잠재력 지표의 유효성을 검증합니다.
    """
    print("\n=== 개선 잠재력 지표 검증 ===")
    
    # 1. 시계열 상관관계 검증
    correlation_results = {}
    
    for year in sorted(bds_data['period'].unique()):
        year_data = bds_data[bds_data['period'] == year].copy()
        
        # 각 지역별 개선 잠재력 계산
        year_data['improvement_potential'] = year_data.apply(lambda row: calculate_single_improvement_potential(row), axis=1)
        
        # BDS와 개선 잠재력의 상관관계
        correlation = year_data['BDS'].corr(year_data['improvement_potential'])
        correlation_results[year] = correlation
    
    avg_correlation = np.mean(list(correlation_results.values()))
    print(f"BDS와 개선 잠재력의 평균 상관계수: {avg_correlation:.4f}")
    
    if abs(avg_correlation) > 0.5:
        print("✅ 강한 상관관계 - 지표가 유효함")
    elif abs(avg_correlation) > 0.3:
        print("⚠️ 중간 상관관계 - 지표가 부분적으로 유효함")
    else:
        print("❌ 약한 상관관계 - 지표 개선 필요")
    
    # 2. 예측 정확도 검증 (과거 데이터 기반)
    validation_results = validate_prediction_accuracy(bds_data)
    
    return {
        'correlation_results': correlation_results,
        'avg_correlation': avg_correlation,
        'validation_results': validation_results
    }

def calculate_single_improvement_potential(row):
    """단일 행에 대한 개선 잠재력 계산"""
    pc_grdp_potential = max(0, (50000 - row['pc_grdp']) / 50000)
    pop_growth_potential = max(0, (1.0 - row['pop_growth']) / 1.0)
    elderly_rate_potential = max(0, (row['elderly_rate'] - 10) / 20)
    fiscal_indep_potential = max(0, (50 - row['fiscal_indep']) / 50)
    
    return (pc_grdp_potential + pop_growth_potential + elderly_rate_potential + fiscal_indep_potential) / 4

def validate_prediction_accuracy(bds_data: pd.DataFrame):
    """예측 정확도 검증 (현실적인 모델)"""
    print("\n=== 예측 정확도 검증 (현실적인 모델) ===")
    
    # 2019년 데이터로 2022년 예측하고 실제와 비교
    base_year = 2019
    target_year = 2022
    
    base_data = bds_data[bds_data['period'] == base_year].copy()
    actual_data = bds_data[bds_data['period'] == target_year].copy()
    
    # 현실적인 예측 모델
    predictions = []
    actuals = []
    regions = []
    prediction_details = []
    
    # 과거 패턴 분석
    pattern_analysis = {}
    for _, base_row in base_data.iterrows():
        region = base_row['region_name']
        region_data = bds_data[bds_data['region_name'] == region].copy()
        region_data = region_data.sort_values('period')
        
        # 2016-2019년 패턴 분석
        recent_data = region_data[(region_data['period'] >= 2016) & (region_data['period'] <= 2019)]
        if len(recent_data) >= 2:
            changes = []
            for i in range(1, len(recent_data)):
                prev_bds = recent_data.iloc[i-1]['BDS']
                curr_bds = recent_data.iloc[i]['BDS']
                if abs(prev_bds) > 0.001:
                    change_rate = (curr_bds - prev_bds) / abs(prev_bds)
                    changes.append(change_rate)
            
            if changes:
                pattern_analysis[region] = np.mean(changes)
            else:
                pattern_analysis[region] = 0.01  # 기본값
        else:
            pattern_analysis[region] = 0.01  # 기본값
    
    for _, base_row in base_data.iterrows():
        region = base_row['region_name']
        actual_row = actual_data[actual_data['region_name'] == region]
        
        if not actual_row.empty:
            actual_row = actual_row.iloc[0]
            
            # 현실적인 예측 모델
            current_bds = base_row['BDS']
            
            # 1. 과거 패턴 기반 예측
            historical_trend = pattern_analysis.get(region, 0.01)
            
            # 2. 개선 잠재력 기반 조정
            improvement_potential = calculate_single_improvement_potential(base_row)
            potential_adjustment = improvement_potential * 0.015  # 더 보수적으로
            
            # 3. 현재 BDS 수준에 따른 조정
            bds_adjustment = max(0, (0.3 - abs(current_bds)) / 0.3) * 0.01
            
            # 4. 최종 예측 변화율
            total_change_rate = historical_trend + potential_adjustment + bds_adjustment
            
            # 5. 3년간의 복리 효과 (더 보수적으로)
            years_passed = target_year - base_year
            predicted_change = current_bds * ((1 + total_change_rate) ** years_passed - 1)
            
            actual_change = actual_row['BDS'] - base_row['BDS']
            
            predictions.append(predicted_change)
            actuals.append(actual_change)
            regions.append(region)
            
            prediction_details.append({
                'region': region,
                'current_bds': current_bds,
                'historical_trend': historical_trend,
                'improvement_potential': improvement_potential,
                'predicted_change': predicted_change,
                'actual_change': actual_change,
                'total_change_rate': total_change_rate
            })
    
    # 예측 정확도 계산
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    print(f"현실적인 모델 예측-실제 상관계수: {correlation:.4f}")
    print(f"평균절대오차(MAE): {mae:.4f}")
    print(f"평균제곱오차(MSE): {mse:.4f}")
    
    if correlation > 0.6:
        print("✅ 높은 예측 정확도")
    elif correlation > 0.3:
        print("⚠️ 중간 예측 정확도")
    else:
        print("❌ 낮은 예측 정확도")
    
    # 상세 분석 출력
    print("\n=== 지역별 예측 상세 분석 (현실적 모델) ===")
    for detail in prediction_details:
        print(f"{detail['region']}: 예측 {detail['predicted_change']:+.3f}, 실제 {detail['actual_change']:+.3f}, 변화율 {detail['total_change_rate']:.3f}")
    
    return {
        'correlation': correlation,
        'mae': mae,
        'mse': mse,
        'predictions': predictions,
        'actuals': actuals,
        'regions': regions,
        'prediction_details': prediction_details
    }

def calculate_investment_impact(priority_analysis: list, investment_budget: float = 1000):
    """
    투자 금액에 따른 예상 성과를 계산합니다.
    """
    print(f"\n=== 투자 효과 분석 (예산: {investment_budget:,.0f}억원) ===")
    
    investment_results = []
    
    for region_data in priority_analysis:
        region = region_data['region']
        potential = region_data['improvement_potential']
        current_bds = region_data['bds_score']
        
        # 지역별 투자 배분 (우선순위 점수 기반)
        investment_share = region_data['overall_priority'] / sum([r['overall_priority'] for r in priority_analysis])
        allocated_budget = investment_budget * investment_share
        
        # 투자 효과 계산 (경험적 공식)
        # 가정: 100억원 투자 시 개선 잠재력의 10% 실현
        investment_effect = (allocated_budget / 100) * potential * 0.1
        expected_bds_improvement = investment_effect
        
        # 경제적 효과 추정
        population = get_estimated_population(region)  # 추정 인구
        
        # 1인당 지역내총생산 개선 효과
        current_grdp = region_data['pc_grdp']
        grdp_improvement_rate = investment_effect * 0.05  # 5% 개선 가정
        expected_grdp = current_grdp * (1 + grdp_improvement_rate)
        grdp_increase = expected_grdp - current_grdp
        
        # 총 경제 효과 (인구 × 1인당 증가액)
        total_economic_effect = population * grdp_increase
        
        # ROI 계산
        roi = (total_economic_effect / (allocated_budget * 100000000)) * 100  # 억원을 원으로 변환
        
        investment_results.append({
            'region': region,
            'allocated_budget': allocated_budget,
            'expected_bds_improvement': expected_bds_improvement,
            'current_grdp': current_grdp,
            'expected_grdp': expected_grdp,
            'grdp_increase': grdp_increase,
            'total_economic_effect': total_economic_effect,
            'roi': roi,
            'population': population
        })
        
        print(f"\n{region}:")
        print(f"  투자 배분: {allocated_budget:.0f}억원")
        print(f"  예상 BDS 개선: {expected_bds_improvement:+.3f}")
        print(f"  1인당 지역내총생산: {current_grdp:,.0f}원 → {expected_grdp:,.0f}원 (+{grdp_increase:,.0f}원)")
        print(f"  총 경제 효과: {total_economic_effect/100000000:,.0f}억원")
        print(f"  투자수익률(ROI): {roi:.1f}%")
    
    return investment_results

def get_estimated_population(region_name: str) -> int:
    """지역별 추정 인구 (2022년 기준, 천명 단위)"""
    population_data = {
        '전라북도': 1780000,
        '경상북도': 2630000,
        '전라남도': 1830000,
        '강원도': 1520000,
        '경상남도': 3310000,
        '서울특별시': 9720000,
        '부산광역시': 3350000,
        '대구광역시': 2410000,
        '인천광역시': 2960000,
        '광주광역시': 1450000,
        '대전광역시': 1460000,
        '울산광역시': 1130000,
        '세종특별자치시': 370000,
        '경기도': 13490000,
        '충청북도': 1600000,
        '충청남도': 2130000,
        '제주특별자치도': 680000
    }
    return population_data.get(region_name, 1000000)  # 기본값 100만명

def create_validation_visualization(validation_results: dict, investment_results: list):
    """
    검증 결과와 투자 효과를 시각화합니다.
    """
    # 1. 예측 정확도 시각화
    predictions = validation_results['predictions']
    actuals = validation_results['actuals']
    regions = validation_results['regions']
    
    fig1 = go.Figure()
    
    # 산점도
    fig1.add_trace(go.Scatter(
        x=predictions,
        y=actuals,
        mode='markers+text',
        text=regions,
        textposition="top center",
        name='예측 vs 실제',
        hovertemplate='<b>%{text}</b><br>예측: %{x:.3f}<br>실제: %{y:.3f}<br><extra></extra>'
    ))
    
    # 완벽한 예측선 (y=x)
    min_val = min(min(predictions), min(actuals))
    max_val = max(max(predictions), max(actuals))
    fig1.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='완벽한 예측',
        line=dict(dash='dash', color='red')
    ))
    
    fig1.update_layout(
        title=f'예측 정확도 검증<br><sub>상관계수: {validation_results["correlation"]:.3f} | MAE: {validation_results["mae"]:.3f}</sub>',
        xaxis_title='예측된 BDS 변화',
        yaxis_title='실제 BDS 변화',
        height=500
    )
    
    # 2. 투자 효과 시각화
    df_investment = pd.DataFrame(investment_results)
    
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=['투자 배분', 'ROI', '경제 효과', 'BDS 개선'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 투자 배분
    fig2.add_trace(go.Bar(
        x=df_investment['region'],
        y=df_investment['allocated_budget'],
        name='투자 배분',
        hovertemplate='<b>%{x}</b><br>투자액: %{y:.0f}억원<br><extra></extra>'
    ), row=1, col=1)
    
    # ROI
    fig2.add_trace(go.Bar(
        x=df_investment['region'],
        y=df_investment['roi'],
        name='ROI',
        hovertemplate='<b>%{x}</b><br>ROI: %{y:.1f}%<br><extra></extra>'
    ), row=1, col=2)
    
    # 경제 효과
    fig2.add_trace(go.Bar(
        x=df_investment['region'],
        y=df_investment['total_economic_effect'] / 100000000,  # 억원 단위
        name='경제 효과',
        hovertemplate='<b>%{x}</b><br>경제효과: %{y:.0f}억원<br><extra></extra>'
    ), row=2, col=1)
    
    # BDS 개선
    fig2.add_trace(go.Bar(
        x=df_investment['region'],
        y=df_investment['expected_bds_improvement'],
        name='BDS 개선',
        hovertemplate='<b>%{x}</b><br>BDS 개선: %{y:+.3f}<br><extra></extra>'
    ), row=2, col=2)
    
    fig2.update_layout(
        title='투자 효과 분석<br><sub>1000억원 투자 시 예상 성과</sub>',
        height=800,
        showlegend=False
    )
    
    return fig1, fig2

def create_scenario_comparison_visualization(scenario_simulations: dict, target_regions: list):
    """
    다양한 시나리오의 시뮬레이션 결과를 비교 시각화합니다.
    """
    # 최종 연도 데이터 추출
    final_year = 2027  # 시뮬레이션 최종 연도
    scenario_comparison = []
    
    for scenario_name, scenario_data in scenario_simulations.items():
        scenario_df = scenario_data['data']
        final_data = scenario_df[scenario_df['period'] == final_year]
        
        for region in target_regions:
            region_data = final_data[final_data['region_name'] == region]
            if not region_data.empty:
                scenario_comparison.append({
                    'scenario': scenario_data['name'],
                    'region': region,
                    'bds': region_data['BDS'].iloc[0],
                    'improvement_rate': scenario_data['improvement_rate']
                })
    
    comparison_df = pd.DataFrame(scenario_comparison)
    
    # 시나리오별 비교 차트
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['시나리오별 BDS 비교', '개선률별 효과', '지역별 시나리오 비교', '시나리오별 평균 개선'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. 시나리오별 BDS 비교
    for scenario in comparison_df['scenario'].unique():
        scenario_data = comparison_df[comparison_df['scenario'] == scenario]
        fig.add_trace(go.Bar(
            x=scenario_data['region'],
            y=scenario_data['bds'],
            name=scenario,
            hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>BDS: %{y:.3f}<br><extra></extra>'
        ), row=1, col=1)
    
    # 2. 개선률별 효과
    improvement_summary = comparison_df.groupby('improvement_rate')['bds'].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=improvement_summary['improvement_rate'] * 100,
        y=improvement_summary['bds'],
        mode='lines+markers',
        name='개선률별 평균 BDS',
        hovertemplate='개선률: %{x:.1f}%<br>평균 BDS: %{y:.3f}<br><extra></extra>'
    ), row=1, col=2)
    
    # 3. 지역별 시나리오 비교
    for region in target_regions:
        region_data = comparison_df[comparison_df['region'] == region]
        fig.add_trace(go.Bar(
            x=region_data['scenario'],
            y=region_data['bds'],
            name=region,
            hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>BDS: %{y:.3f}<br><extra></extra>'
        ), row=2, col=1)
    
    # 4. 시나리오별 평균 개선
    scenario_avg = comparison_df.groupby('scenario')['bds'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=scenario_avg['scenario'],
        y=scenario_avg['bds'],
        name='시나리오별 평균',
        hovertemplate='<b>%{x}</b><br>평균 BDS: %{y:.3f}<br><extra></extra>'
    ), row=2, col=2)
    
    fig.update_layout(
        title='다양한 시나리오 시뮬레이션 비교<br><sub>절차: 1) 보수적(3%) 2) 중간(5%) 3) 적극적(8%) 4) 차별화(6%) 시나리오 비교</sub>',
        height=800,
        showlegend=False
    )
    
    return fig

def create_realistic_simulation(bds_data: pd.DataFrame, target_regions: list):
    """
    실제 데이터 패턴을 기반으로 한 현실적인 시뮬레이션을 생성합니다.
    """
    print("\n=== 현실적인 시뮬레이션 모델 생성 ===")
    
    # 과거 데이터 패턴 분석
    pattern_analysis = analyze_historical_patterns(bds_data, target_regions)
    
    # 최신 연도 데이터
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    
    # 시뮬레이션 데이터 생성
    simulation_data = []
    
    for year in range(latest_year + 1, latest_year + 6):
        year_data = latest_data.copy()
        year_data['period'] = year
        
        # 대상 지역들의 현실적인 BDS 개선
        for region in target_regions:
            mask = year_data['region_name'] == region
            if mask.any():
                region_data = year_data[mask].iloc[0]
                current_bds = region_data['BDS']
                
                # 과거 패턴 기반 개선률 계산
                historical_trend = pattern_analysis.get(region, 0.02)  # 기본 2%
                
                # 개선 잠재력 기반 조정
                improvement_potential = calculate_single_improvement_potential(region_data)
                potential_adjustment = improvement_potential * 0.03  # 최대 3% 추가
                
                # 현재 BDS 수준에 따른 조정 (낮을수록 더 큰 개선)
                bds_level_adjustment = max(0, (0.5 - abs(current_bds)) / 0.5) * 0.02
                
                # 최종 개선률
                total_improvement_rate = historical_trend + potential_adjustment + bds_level_adjustment
                
                # 체감 효과 (시간이 지날수록 개선 효과 감소)
                years_passed = year - latest_year
                diminishing_factor = 1 / (1 + years_passed * 0.2)
                final_improvement_rate = total_improvement_rate * diminishing_factor
                
                # BDS 개선 적용 (더 보수적으로)
                improvement_factor = 1 + final_improvement_rate
                year_data.loc[mask, 'BDS'] *= improvement_factor
        
        simulation_data.append(year_data)
    
    # 결과 결합
    result = pd.concat([bds_data, pd.concat(simulation_data, ignore_index=True)], 
                      ignore_index=True)
    
    return result, target_regions, pattern_analysis

def analyze_historical_patterns(bds_data: pd.DataFrame, target_regions: list):
    """
    과거 데이터의 패턴을 분석하여 각 지역의 개선 트렌드를 계산합니다.
    """
    print("과거 데이터 패턴 분석 중...")
    
    pattern_analysis = {}
    
    for region in target_regions:
        region_data = bds_data[bds_data['region_name'] == region].copy()
        region_data = region_data.sort_values('period')
        
        if len(region_data) >= 3:
            # 최근 3년간의 평균 변화율 계산
            recent_changes = []
            for i in range(1, len(region_data)):
                prev_bds = region_data.iloc[i-1]['BDS']
                curr_bds = region_data.iloc[i]['BDS']
                
                if abs(prev_bds) > 0.001:  # 0이 아닌 경우만
                    change_rate = (curr_bds - prev_bds) / abs(prev_bds)
                    recent_changes.append(change_rate)
            
            if recent_changes:
                # 최근 3년 평균 변화율 (양수면 개선, 음수면 악화)
                avg_change_rate = np.mean(recent_changes[-3:]) if len(recent_changes) >= 3 else np.mean(recent_changes)
                
                # 개선 트렌드가 있으면 그대로 사용, 없으면 기본값
                if avg_change_rate > 0:
                    pattern_analysis[region] = avg_change_rate
                else:
                    # 개선 잠재력 기반 기본 개선률
                    latest_data = region_data.iloc[-1]
                    improvement_potential = calculate_single_improvement_potential(latest_data)
                    pattern_analysis[region] = improvement_potential * 0.02  # 잠재력의 2%
                
                print(f"{region}: 과거 평균 변화율 {avg_change_rate:.4f}, 적용 개선률 {pattern_analysis[region]:.4f}")
            else:
                pattern_analysis[region] = 0.02  # 기본값
        else:
            pattern_analysis[region] = 0.02  # 기본값
    
    return pattern_analysis

def main():
    """
    메인 실행 함수
    """
    print("=== NAVIS 지역발전지수 분석 시작 ===\n")
    
    # 1. NAVIS 데이터 로드
    print("1. NAVIS 데이터 로드 중...")
    navis_data = load_and_clean_navis_data()
    
    if not navis_data:
        print("NAVIS 데이터를 로드할 수 없습니다.")
        return
    
    print(f"로드된 시트: {list(navis_data.keys())}")
    
    # 2. BDS 데이터 로드 (가정: 이미 생성되어 있음)
    print("\n2. BDS 데이터 로드 중...")
    try:
        bds_data = pd.read_csv("outputs_timeseries/bds_timeseries.csv")
        print(f"BDS 데이터 로드 완료: {bds_data.shape}")
    except FileNotFoundError:
        print("BDS 데이터 파일을 찾을 수 없습니다. 먼저 bds_analysis.py를 실행하세요.")
        return
    
    # 3. 지역 균형성 분석
    print("\n3. 지역 균형성 분석 중...")
    balance_analysis = analyze_regional_balance_indicators(bds_data)
    print("균형성 분석 완료")
    
    # 4. NAVIS와 비교
    print("\n4. NAVIS 지수와 비교 분석 중...")
    comparison_results = compare_bds_with_navis(bds_data, navis_data, target_year=2019)
    print("비교 분석 완료")
    
    # 5. 시뮬레이션 생성
    print("\n5. 지역 균형 개선 시뮬레이션 생성 중...")
    
    # 개선 대상 지역 선정
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    target_regions = latest_data.nsmallest(5, 'BDS')['region_name'].tolist()
    
    # 현실적인 시뮬레이션 생성
    simulation_data, target_regions, pattern_analysis = create_realistic_simulation(bds_data, target_regions)
    
    # 다양한 시나리오 시뮬레이션 생성
    print("다양한 시나리오 시뮬레이션 생성 중...")
    scenario_simulations = create_multiple_simulation_scenarios(bds_data, target_regions)
    print("시뮬레이션 생성 완료")
    
    # 6. 투자 우선순위 분석
    print("\n6. 투자 우선순위 분석 중...")
    priority_analysis = analyze_investment_priority(bds_data, target_regions)
    print("투자 우선순위 분석 완료")
    
    # 7. 지표 검증 및 투자 효과 분석
    print("\n7. 지표 검증 및 투자 효과 분석 중...")
    validation_results = validate_improvement_potential(bds_data)
    investment_results = calculate_investment_impact(priority_analysis, investment_budget=1000)
    print("검증 및 투자 효과 분석 완료")
    
    # 8. 시각화 생성
    print("\n8. 시각화 생성 중...")
    
    # 기본 시뮬레이션 시각화
    sim_fig = visualize_simulation(simulation_data, target_regions)
    sim_fig.write_html("outputs_timeseries/regional_balance_simulation.html")
    print("- 기본 시뮬레이션 시각화 저장 완료")
    
    # 시나리오별 시뮬레이션 시각화
    scenario_fig = create_scenario_comparison_visualization(scenario_simulations, target_regions)
    scenario_fig.write_html("outputs_timeseries/scenario_comparison.html")
    print("- 시나리오 비교 시각화 저장 완료")
    
    # 시뮬레이션 대시보드
    fig1, fig2, fig3, fig4 = create_simulation_dashboard(simulation_data, target_regions, balance_analysis)
    fig1.write_html("outputs_timeseries/simulation_trend.html")
    fig2.write_html("outputs_timeseries/simulation_final_comparison.html")
    fig3.write_html("outputs_timeseries/simulation_improvement.html")
    fig4.write_html("outputs_timeseries/simulation_balance_metrics.html")
    print("- 시뮬레이션 대시보드 저장 완료")
    
    # 투자 우선순위 시각화
    priority_fig1, priority_fig2 = create_investment_priority_visualization(priority_analysis)
    priority_fig1.write_html("outputs_timeseries/investment_priority.html")
    priority_fig2.write_html("outputs_timeseries/improvement_potential.html")
    print("- 투자 우선순위 시각화 저장 완료")
    
    # 검증 및 투자 효과 시각화
    validation_fig1, validation_fig2 = create_validation_visualization(validation_results['validation_results'], investment_results)
    validation_fig1.write_html("outputs_timeseries/prediction_validation.html")
    validation_fig2.write_html("outputs_timeseries/investment_impact.html")
    print("- 검증 및 투자 효과 시각화 저장 완료")
    
    # 요약 시각화
    sum_fig1, sum_fig2 = create_summary_visualization(bds_data, target_regions)
    sum_fig1.write_html("outputs_timeseries/current_situation.html")
    sum_fig2.write_html("outputs_timeseries/regional_indicators.html")
    print("- 요약 시각화 저장 완료")
    
    # NAVIS 비교 시각화
    if comparison_results:
        comp_fig = visualize_comparison(comparison_results, target_year=2019)
        if comp_fig:
            comp_fig.write_html("outputs_timeseries/bds_navis_comparison.html")
            print("- NAVIS 비교 시각화 저장 완료")
    
    # 9. 보고서 생성
    print("\n9. 분석 보고서 생성 중...")
    report = generate_analysis_report(bds_data, navis_data, comparison_results, balance_analysis)
    
    # 투자 우선순위 정보 추가
    report += "\n## 4. 투자 우선순위 분석\n\n"
    for i, item in enumerate(priority_analysis, 1):
        report += f"### {i}순위: {item['region']}\n"
        report += f"- BDS 점수: {item['bds_score']:.3f}\n"
        report += f"- 투자 우선순위 점수: {item['overall_priority']:.3f}\n"
        report += f"- 개선 잠재력: {item['improvement_potential']:.3f}\n"
        report += f"- 1인당 지역내총생산: {item['pc_grdp']:,.0f}원\n"
        report += f"- 인구증가율: {item['pop_growth']:.2f}%\n"
        report += f"- 고령인구비율: {item['elderly_rate']:.1f}%\n"
        report += f"- 재정자립도: {item['fiscal_indep']:.1f}%\n\n"
    
    # 지표 검증 결과 추가
    report += "\n## 5. 지표 검증 결과\n\n"
    report += f"### 개선 잠재력 지표 검증\n"
    report += f"- BDS와 개선 잠재력 평균 상관계수: {validation_results['avg_correlation']:.4f}\n"
    report += f"- 예측 정확도 (상관계수): {validation_results['validation_results']['correlation']:.4f}\n"
    report += f"- 평균절대오차(MAE): {validation_results['validation_results']['mae']:.4f}\n"
    report += f"- 평균제곱오차(MSE): {validation_results['validation_results']['mse']:.4f}\n\n"
    
    # 투자 효과 분석 추가
    report += "\n## 6. 투자 효과 분석 (1000억원 투자 시)\n\n"
    total_roi = sum([r['roi'] for r in investment_results]) / len(investment_results)
    total_economic_effect = sum([r['total_economic_effect'] for r in investment_results])
    
    report += f"### 전체 투자 효과\n"
    report += f"- 평균 투자수익률(ROI): {total_roi:.1f}%\n"
    report += f"- 총 경제 효과: {total_economic_effect/100000000:,.0f}억원\n\n"
    
    for result in investment_results:
        report += f"### {result['region']}\n"
        report += f"- 투자 배분: {result['allocated_budget']:.0f}억원\n"
        report += f"- 예상 BDS 개선: {result['expected_bds_improvement']:+.3f}\n"
        report += f"- 1인당 지역내총생산 증가: {result['grdp_increase']:,.0f}원\n"
        report += f"- 총 경제 효과: {result['total_economic_effect']/100000000:,.0f}억원\n"
        report += f"- 투자수익률(ROI): {result['roi']:.1f}%\n\n"
    
    with open("outputs_timeseries/analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("- 분석 보고서 저장 완료")
    
    # 10. 결과 출력
    print("\n=== 분석 결과 요약 ===")
    print(f"지역 균형성 (변동계수): {balance_analysis['cv']:.4f}")
    print(f"지니계수: {balance_analysis['gini']:.4f}")
    
    print(f"\n=== 지표 검증 결과 ===")
    print(f"개선 잠재력 지표 유효성: {validation_results['avg_correlation']:.4f}")
    print(f"예측 정확도: {validation_results['validation_results']['correlation']:.4f}")
    
    if comparison_results:
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        if valid_results:
            avg_corr = np.mean([r['correlation'] for r in valid_results.values()])
            print(f"NAVIS 지수와의 평균 상관계수: {avg_corr:.4f}")
    
    print(f"\n개선 대상 지역: {', '.join(target_regions)}")
    print(f"\n투자 우선순위:")
    for i, item in enumerate(priority_analysis, 1):
        print(f"  {i}. {item['region']} (우선순위 점수: {item['overall_priority']:.3f})")
    
    print(f"\n=== 투자 효과 (1000억원) ===")
    print(f"평균 ROI: {total_roi:.1f}%")
    print(f"총 경제 효과: {total_economic_effect/100000000:,.0f}억원")
    
    print("\n=== 분석 완료 ===")
    print("생성된 파일들:")
    print("- outputs_timeseries/regional_balance_simulation.html (기본 시뮬레이션)")
    print("- outputs_timeseries/simulation_trend.html (시뮬레이션 추이)")
    print("- outputs_timeseries/simulation_final_comparison.html (최종 비교)")
    print("- outputs_timeseries/simulation_improvement.html (개선 효과)")
    print("- outputs_timeseries/simulation_balance_metrics.html (균형성 지표)")
    print("- outputs_timeseries/investment_priority.html (투자 우선순위)")
    print("- outputs_timeseries/improvement_potential.html (개선 잠재력)")
    print("- outputs_timeseries/prediction_validation.html (예측 검증)")
    print("- outputs_timeseries/investment_impact.html (투자 효과)")
    print("- outputs_timeseries/current_situation.html (현재 상황)")
    print("- outputs_timeseries/regional_indicators.html (지역별 지표)")
    print("- outputs_timeseries/bds_navis_comparison.html (NAVIS 비교)")
    print("- outputs_timeseries/analysis_report.md (분석 보고서)")

if __name__ == "__main__":
    main() 