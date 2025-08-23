import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """데이터를 로드합니다."""
    # BDS 모델 데이터
    bds_df = pd.read_csv('enhanced_bds_model_with_kosis.csv')
    
    # NAVIS 데이터
    navis_df = pd.read_excel('navis_data/1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx', 
                            sheet_name='I지역발전지수(총합)')
    
    # NAVIS 데이터 정리
    navis_clean = navis_df.melt(id_vars=['지역발전지수'], var_name='year', value_name='navis_value')
    navis_clean = navis_clean[navis_clean['지역발전지수'] != '지역발전지수']
    navis_clean['year'] = pd.to_numeric(navis_clean['year'], errors='coerce')
    navis_clean = navis_clean.dropna()
    
    # Geojson 로드
    with open('navis_data/skorea-provinces-2018-geo.json', 'r', encoding='utf-8') as f:
        geojson = json.load(f)
    
    return bds_df, navis_clean, geojson

def create_comprehensive_dashboard():
    """종합 대시보드를 생성합니다."""
    print("KOSIS 데이터 기반 BDS 모델 종합 대시보드 생성 중...")
    
    bds_df, navis_df, geojson = load_data()
    
    # 지역명 매핑
    region_mapping = {
        '서울특별시': '서울',
        '부산광역시': '부산',
        '대구광역시': '대구',
        '인천광역시': '인천',
        '광주광역시': '광주',
        '대전광역시': '대전',
        '울산광역시': '울산',
        '세종특별자치시': '세종',
        '경기도': '경기',
        '강원도': '강원',
        '충청북도': '충북',
        '충청남도': '충남',
        '전라북도': '전북',
        '전라남도': '전남',
        '경상북도': '경북',
        '경상남도': '경남',
        '제주도': '제주'
    }
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'NAVIS vs BDS 상관관계 (2019년까지)',
            '2020년 이후 GDP vs BDS 상관관계',
            '지역별 BDS 성능 등급 (2025년)',
            '시계열 비교: NAVIS vs BDS',
            '2025년 지역별 BDS 예측값',
            'GDP 기반 BDS 검증'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # 1. NAVIS vs BDS 상관관계 (2019년까지)
    validation_data = bds_df[bds_df['year'] <= 2019].copy()
    navis_validation = navis_df.copy()
    
    # 데이터 병합
    merged_data = []
    for _, row in validation_data.iterrows():
        navis_value = navis_validation[
            (navis_validation['지역발전지수'] == row['region']) & 
            (navis_validation['year'] == row['year'])
        ]['navis_value'].values
        
        if len(navis_value) > 0:
            merged_data.append({
                'region': row['region'],
                'year': row['year'],
                'bds_value': row['bds_value'],
                'navis_value': navis_value[0]
            })
    
    merged_df = pd.DataFrame(merged_data)
    
    if len(merged_df) > 0:
        correlation = merged_df['bds_value'].corr(merged_df['navis_value'])
        
        fig.add_trace(
            go.Scatter(
                x=merged_df['navis_value'],
                y=merged_df['bds_value'],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                name=f'상관관계: {correlation:.3f}',
                hovertemplate='<b>%{text}</b><br>NAVIS: %{x:.2f}<br>BDS: %{y:.2f}<extra></extra>',
                text=[f"{row['region']} ({row['year']})" for _, row in merged_df.iterrows()]
            ),
            row=1, col=1
        )
    
    # 2. 2020년 이후 GDP vs BDS 상관관계
    recent_data = bds_df[bds_df['year'] >= 2020].copy()
    recent_data = recent_data.dropna(subset=['gdp_value'])
    
    if len(recent_data) > 0:
        gdp_correlation = recent_data['bds_value'].corr(recent_data['gdp_value'])
        
        fig.add_trace(
            go.Scatter(
                x=recent_data['gdp_value'],
                y=recent_data['bds_value'],
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.6),
                name=f'GDP 상관관계: {gdp_correlation:.3f}',
                hovertemplate='<b>%{text}</b><br>GDP: %{x:,.0f}<br>BDS: %{y:.2f}<extra></extra>',
                text=[f"{row['region']} ({row['year']})" for _, row in recent_data.iterrows()]
            ),
            row=1, col=2
        )
    
    # 3. 지역별 BDS 성능 등급 (2025년)
    latest_data = bds_df[bds_df['year'] == 2025].copy()
    
    # 성능 등급 계산
    latest_data['performance_grade'] = pd.cut(
        latest_data['bds_value'],
        bins=[0, 45, 50, 55, 100],
        labels=['D', 'C', 'B', 'A']
    )
    
    grade_counts = latest_data['performance_grade'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=grade_counts.index,
            y=grade_counts.values,
            marker_color=['red', 'orange', 'yellow', 'green'],
            name='성능 등급',
            hovertemplate='등급: %{x}<br>지역 수: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. 시계열 비교: NAVIS vs BDS
    # 서울 데이터로 예시
    seoul_bds = bds_df[bds_df['region'] == '서울특별시'].copy()
    seoul_navis = navis_df[navis_df['지역발전지수'] == '서울특별시'].copy()
    
    fig.add_trace(
        go.Scatter(
            x=seoul_bds['year'],
            y=seoul_bds['bds_value'],
            mode='lines+markers',
            name='BDS (서울)',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=seoul_navis['year'],
            y=seoul_navis['navis_value'],
            mode='lines+markers',
            name='NAVIS (서울)',
            line=dict(color='red', width=2)
        ),
        row=2, col=2
    )
    
    # 5. 2025년 지역별 BDS 예측값
    fig.add_trace(
        go.Bar(
            x=latest_data['region'],
            y=latest_data['bds_value'],
            marker_color='lightblue',
            name='2025년 BDS',
            hovertemplate='<b>%{x}</b><br>BDS: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. GDP 기반 BDS 검증
    if len(recent_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=recent_data['year'],
                y=recent_data['bds_value'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=recent_data['gdp_value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="GDP (십억원)")
                ),
                name='BDS vs GDP',
                hovertemplate='<b>%{text}</b><br>연도: %{x}<br>BDS: %{y:.2f}<br>GDP: %{marker.color:,.0f}<extra></extra>',
                text=recent_data['region']
            ),
            row=3, col=2
        )
    
    # 레이아웃 업데이트
    fig.update_layout(
        title={
            'text': 'KOSIS 데이터 기반 향상된 BDS 모델 종합 분석',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1200,
        width=1400,
        showlegend=True,
        template='plotly_white'
    )
    
    # 축 레이블 업데이트
    fig.update_xaxes(title_text="NAVIS 값", row=1, col=1)
    fig.update_yaxes(title_text="BDS 값", row=1, col=1)
    
    fig.update_xaxes(title_text="GDP (십억원)", row=1, col=2)
    fig.update_yaxes(title_text="BDS 값", row=1, col=2)
    
    fig.update_xaxes(title_text="성능 등급", row=2, col=1)
    fig.update_yaxes(title_text="지역 수", row=2, col=1)
    
    fig.update_xaxes(title_text="연도", row=2, col=2)
    fig.update_yaxes(title_text="지수 값", row=2, col=2)
    
    fig.update_xaxes(title_text="지역", row=3, col=1)
    fig.update_yaxes(title_text="BDS 값", row=3, col=1)
    
    fig.update_xaxes(title_text="연도", row=3, col=2)
    fig.update_yaxes(title_text="BDS 값", row=3, col=2)
    
    # HTML 파일로 저장
    fig.write_html('kosis_bds_comprehensive_dashboard.html')
    print("대시보드 생성 완료: kosis_bds_comprehensive_dashboard.html")
    
    return fig

def create_geojson_comparison():
    """Geojson 기반 지역별 비교 시각화를 생성합니다."""
    print("Geojson 기반 지역별 비교 시각화 생성 중...")
    
    bds_df, navis_df, geojson = load_data()
    
    # 2025년 BDS 데이터
    latest_bds = bds_df[bds_df['year'] == 2025].copy()
    
    # 지역명 매핑
    region_mapping = {
        '서울특별시': '서울',
        '부산광역시': '부산',
        '대구광역시': '대구',
        '인천광역시': '인천',
        '광주광역시': '광주',
        '대전광역시': '대전',
        '울산광역시': '울산',
        '세종특별자치시': '세종',
        '경기도': '경기',
        '강원도': '강원',
        '충청북도': '충북',
        '충청남도': '충남',
        '전라북도': '전북',
        '전라남도': '전남',
        '경상북도': '경북',
        '경상남도': '경남',
        '제주도': '제주'
    }
    
    # Geojson에 BDS 데이터 추가
    for feature in geojson['features']:
        region_name = feature['properties']['name']
        if region_name in region_mapping:
            navis_name = region_mapping[region_name]
            bds_value = latest_bds[latest_bds['region'] == navis_name]['bds_value'].values
            if len(bds_value) > 0:
                feature['properties']['bds_value'] = float(bds_value[0])
            else:
                feature['properties']['bds_value'] = 0
    
    # Choropleth 맵 생성
    fig = go.Figure()
    
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=[feature['properties']['name'] for feature in geojson['features']],
        z=[feature['properties'].get('bds_value', 0) for feature in geojson['features']],
        colorscale='Viridis',
        marker_opacity=0.7,
        marker_line_width=1,
        colorbar_title="BDS 값 (2025년)",
        hovertemplate="<b>%{location}</b><br>BDS: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title={
            'text': '2025년 지역별 BDS 예측값 (KOSIS 데이터 기반)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=36.5, lon=127.5),
            zoom=6
        ),
        height=600,
        width=800
    )
    
    # HTML 파일로 저장
    fig.write_html('kosis_bds_geojson_map.html')
    print("Geojson 맵 생성 완료: kosis_bds_geojson_map.html")
    
    return fig

def main():
    """메인 실행 함수"""
    print("=== KOSIS 데이터 기반 BDS 모델 시각화 생성 ===")
    
    try:
        # 종합 대시보드 생성
        create_comprehensive_dashboard()
        
        # Geojson 비교 시각화 생성
        create_geojson_comparison()
        
        print("\n=== 시각화 생성 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
