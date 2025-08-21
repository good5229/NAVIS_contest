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
            year_data = df[['지역명', str(target_year)]].copy()
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
                showlegend=False
            ),
            row=1, col=i
        )
        
        # 상관계수 표시
        corr = result['correlation']
        fig.add_annotation(
            x=0.5, y=0.95,
            xref=f'x{i}', yref=f'y{i}',
            text=f'상관계수: {corr:.3f}<br>순위상관: {result["rank_correlation"]:.3f}',
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
                showlegend=False
            ),
            row=2, col=i
        )
        
        # 축 레이블
        fig.update_xaxes(title_text="BDS", row=1, col=i)
        fig.update_yaxes(title_text=f"{navis_type}", row=1, col=i)
        fig.update_xaxes(title_text="상관관계 지표", row=2, col=i)
        fig.update_yaxes(title_text="상관계수", row=2, col=i, range=[0, 1])
    
    fig.update_layout(
        title=f'BDS와 NAVIS 지역발전지수 비교 분석 ({target_year}년)',
        height=800,
        showlegend=False
    )
    
    return fig

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
            showlegend=True
        ))
    
    # 기존 라인들의 범례 숨기기 (대상 지역만 표시)
    for trace in fig.data:
        if trace.name not in [f'{region} (개선대상)' for region in target_regions]:
            trace.showlegend = False
    
    fig.update_layout(
        height=600,
        title={
            'text': '지역 균형 개선 시뮬레이션<br><sub>빨간 점선: 개선 대상 지역 (연 10% BDS 향상)</sub>',
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
            showlegend=True
        ))
    
    # 기존 라인들의 범례 숨기기
    for trace in fig1.data:
        if trace.name not in [f'{region} (개선대상)' for region in target_regions]:
            trace.showlegend = False
    
    # 2. 최종 연도 비교 차트
    final_year = simulation_data['period'].max()
    final_data = simulation_data[simulation_data['period'] == final_year].copy()
    final_data['is_target'] = final_data['region_name'].isin(target_regions)
    
    fig2 = px.bar(final_data, x='region_name', y='BDS', color='is_target',
                  title=f'{final_year}년 지역별 BDS 점수',
                  labels={'region_name': '지역', 'BDS': 'BDS 점수', 'is_target': '개선대상'},
                  color_discrete_map={True: 'red', False: '#1f77b4'})
    
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
    
    # 4. 균형성 지표 변화 차트
    years = sorted(simulation_data['period'].unique())
    balance_metrics = []
    
    for year in years:
        year_data = simulation_data[simulation_data['period'] == year]
        bds_values = year_data['BDS'].values
        
        metrics = {
            'year': year,
            'mean': np.mean(bds_values),
            'std': np.std(bds_values),
            'cv': np.std(bds_values) / abs(np.mean(bds_values)) if np.mean(bds_values) != 0 else 0,
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
    
    fig4.add_trace(go.Scatter(x=balance_df['year'], y=balance_df['mean'], name='평균'), row=1, col=1)
    fig4.add_trace(go.Scatter(x=balance_df['year'], y=balance_df['std'], name='표준편차'), row=1, col=2)
    fig4.add_trace(go.Scatter(x=balance_df['year'], y=balance_df['cv'], name='변동계수'), row=2, col=1)
    fig4.add_trace(go.Scatter(x=balance_df['year'], y=balance_df['gini'], name='지니계수'), row=2, col=2)
    
    fig4.update_layout(title='균형성 지표 변화 추이', height=600, showlegend=False)
    
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
    fig2.add_trace(go.Bar(x=other_data['region_name'], y=other_data['pc_grdp'], 
                          name='기타 지역', marker_color='#1f77b4'), row=1, col=1)
    fig2.add_trace(go.Bar(x=target_data['region_name'], y=target_data['pc_grdp'], 
                          name='개선대상', marker_color='red'), row=1, col=1)
    
    # 인구증가율
    fig2.add_trace(go.Bar(x=other_data['region_name'], y=other_data['pop_growth'], 
                          name='기타 지역', marker_color='#1f77b4', showlegend=False), row=1, col=2)
    fig2.add_trace(go.Bar(x=target_data['region_name'], y=target_data['pop_growth'], 
                          name='개선대상', marker_color='red', showlegend=False), row=1, col=2)
    
    # 고령인구비율
    fig2.add_trace(go.Bar(x=other_data['region_name'], y=other_data['elderly_rate'], 
                          name='기타 지역', marker_color='#1f77b4', showlegend=False), row=2, col=1)
    fig2.add_trace(go.Bar(x=target_data['region_name'], y=target_data['elderly_rate'], 
                          name='개선대상', marker_color='red', showlegend=False), row=2, col=1)
    
    # 재정자립도
    fig2.add_trace(go.Bar(x=other_data['region_name'], y=other_data['fiscal_indep'], 
                          name='기타 지역', marker_color='#1f77b4', showlegend=False), row=2, col=2)
    fig2.add_trace(go.Bar(x=target_data['region_name'], y=target_data['fiscal_indep'], 
                          name='개선대상', marker_color='red', showlegend=False), row=2, col=2)
    
    fig2.update_layout(title='지역별 4개 핵심 지표 비교', height=800)
    
    return fig1, fig2

def create_regional_balance_simulation(bds_data: pd.DataFrame, 
                                     target_regions: list = None,
                                     improvement_rate: float = 0.1,
                                     simulation_years: int = 5):
    """
    지역 균형 개선 시뮬레이션을 생성합니다.
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
        
        # 대상 지역들의 BDS 개선
        for region in target_regions:
            mask = year_data['region_name'] == region
            if mask.any():
                years_passed = year - latest_year
                improvement_factor = (1 + improvement_rate) ** years_passed
                year_data.loc[mask, 'BDS'] *= improvement_factor
        
        simulation_data.append(year_data)
    
    # 결과 결합
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
    
    # 다양한 균형성 지표 계산
    analysis = {
        'year': latest_year,
        'mean': np.mean(bds_values),
        'std': np.std(bds_values),
        'cv': np.std(bds_values) / np.mean(bds_values),  # 변동계수
        'gini': calculate_gini_coefficient(bds_values),
        'max_min_ratio': np.max(bds_values) / np.min(bds_values),
        'top_bottom_ratio': np.percentile(bds_values, 80) / np.percentile(bds_values, 20),
        'iqr': np.percentile(bds_values, 75) - np.percentile(bds_values, 25),
        'skewness': stats.skew(bds_values),
        'kurtosis': stats.kurtosis(bds_values)
    }
    
    return analysis

def calculate_gini_coefficient(values: np.ndarray) -> float:
    """지니계수 계산"""
    sorted_values = np.sort(values)
    n = len(sorted_values)
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
    simulation_data, target_regions = create_regional_balance_simulation(bds_data)
    print("시뮬레이션 생성 완료")
    
    # 6. 시각화 생성
    print("\n6. 시각화 생성 중...")
    
    # 기본 시뮬레이션 시각화
    sim_fig = visualize_simulation(simulation_data, target_regions)
    sim_fig.write_html("outputs_timeseries/regional_balance_simulation.html")
    print("- 기본 시뮬레이션 시각화 저장 완료")
    
    # 시뮬레이션 대시보드
    fig1, fig2, fig3, fig4 = create_simulation_dashboard(simulation_data, target_regions, balance_analysis)
    fig1.write_html("outputs_timeseries/simulation_trend.html")
    fig2.write_html("outputs_timeseries/simulation_final_comparison.html")
    fig3.write_html("outputs_timeseries/simulation_improvement.html")
    fig4.write_html("outputs_timeseries/simulation_balance_metrics.html")
    print("- 시뮬레이션 대시보드 저장 완료")
    
    # 요약 시각화
    sum_fig1, sum_fig2 = create_summary_visualization(bds_data, target_regions)
    sum_fig1.write_html("outputs_timeseries/current_situation.html")
    sum_fig2.write_html("outputs_timeseries/regional_indicators.html")
    print("- 요약 시각화 저장 완료")
    
    # 비교 시각화
    if comparison_results:
        comp_fig = visualize_comparison(comparison_results, target_year=2019)
        if comp_fig:
            comp_fig.write_html("outputs_timeseries/bds_navis_comparison.html")
            print("- NAVIS 비교 시각화 저장 완료")
    
    # 7. 보고서 생성
    print("\n7. 분석 보고서 생성 중...")
    report = generate_analysis_report(bds_data, navis_data, comparison_results, balance_analysis)
    
    with open("outputs_timeseries/analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("- 분석 보고서 저장 완료")
    
    # 8. 결과 출력
    print("\n=== 분석 결과 요약 ===")
    print(f"지역 균형성 (변동계수): {balance_analysis['cv']:.4f}")
    print(f"지니계수: {balance_analysis['gini']:.4f}")
    
    if comparison_results:
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        if valid_results:
            avg_corr = np.mean([r['correlation'] for r in valid_results.values()])
            print(f"NAVIS 지수와의 평균 상관계수: {avg_corr:.4f}")
    
    print(f"\n개선 대상 지역: {', '.join(target_regions)}")
    
    print("\n=== 분석 완료 ===")
    print("생성된 파일들:")
    print("- outputs_timeseries/regional_balance_simulation.html (기본 시뮬레이션)")
    print("- outputs_timeseries/simulation_trend.html (시뮬레이션 추이)")
    print("- outputs_timeseries/simulation_final_comparison.html (최종 비교)")
    print("- outputs_timeseries/simulation_improvement.html (개선 효과)")
    print("- outputs_timeseries/simulation_balance_metrics.html (균형성 지표)")
    print("- outputs_timeseries/current_situation.html (현재 상황)")
    print("- outputs_timeseries/regional_indicators.html (지역별 지표)")
    print("- outputs_timeseries/bds_navis_comparison.html (NAVIS 비교)")
    print("- outputs_timeseries/analysis_report.md (분석 보고서)")

if __name__ == "__main__":
    main() 