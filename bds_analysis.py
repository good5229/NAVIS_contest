# -*- coding: utf-8 -*-
"""
BDS 타임시리즈 + 지도(애니메이션) — 메타 호출 없이 Param/getList만 사용
- 최근 10년 단위로 시도별 BDS 산출(PCA/동일/사용자 가중)
- Plotly 코로플레스(연도 슬라이더) HTML 출력
- GeoJSON의 시도명 필드가 '한글/영문' 어느 쪽이든 자동 매칭 (한글이면 매핑 금지)

준비:
1) pip install requests pandas numpy scikit-learn python-dotenv plotly
2) .env 또는 OS 환경변수에 KOSIS_API_KEY 등록
3) 시도 경계 GeoJSON 경로 지정(아래 __main__의 GEOJSON_PATH 수정)
"""

import os, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------------- 설정 ----------------
load_dotenv()
API_KEY = os.getenv("KOSIS_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 KOSIS_API_KEY가 필요합니다. (.env에 추가하세요)")

OUTDIR = Path("./outputs_timeseries")
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE_PARAM = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

# 표 정의(공식 표ID, ITM_NM 키워드, 연간: prdSe='Y')
TABLES = {
    "pc_grdp":      {"org":"101","tbl":"DT_1C86",     "keyword":"1인당 지역내총생산","prdSe":"Y"},
    "pop_growth":   {"org":"101","tbl":"DT_1YL20621","keyword":"인구증가율","prdSe":"Y"},
    "elderly_rate": {"org":"101","tbl":"DT_1YL20631","keyword":"고령인구비율","prdSe":"Y"},
    "fiscal_indep": {"org":"101","tbl":"DT_1YL20921","keyword":"재정자립도","prdSe":"Y"},
}

# 이름 정규화(GeoJSON/표 간 표기차 흡수)
def normalize_region_kor(s: str) -> str:
    s = str(s).strip()
    rep = {
        "세종시":"세종특별자치시",
        "제주도":"제주특별자치도",
        "전북특별자치도":"전라북도",   # 오래된 GeoJSON 호환용
        "강원특별자치도":"강원도",
        "전남":"전라남도","전북":"전라북도",
        "경남":"경상남도","경북":"경상북도",
        "충남":"충청남도","충북":"충청북도",
    }
    return rep.get(s, s)

KOR2ENG = {
    "서울특별시":"Seoul","부산광역시":"Busan","대구광역시":"Daegu","인천광역시":"Incheon",
    "광주광역시":"Gwangju","대전광역시":"Daejeon","울산광역시":"Ulsan",
    "세종특별자치시":"Sejong","경기도":"Gyeonggi-do","강원도":"Gangwon-do",
    "충청북도":"Chungcheongbuk-do","충청남도":"Chungcheongnam-do",
    "전라북도":"Jeollabuk-do","전라남도":"Jeollanam-do",
    "경상북도":"Gyeongsangbuk-do","경상남도":"Gyeongsangnam-do","제주특별자치도":"Jeju-do"
}

# ----------- KOSIS 호출(Param/getList 전용, JSON 방어 포함) -----------
def _get_json(url: str, params: Dict, retry=3, sleep=0.6):
    """
    일부 상황에서 서버가 HTML/문자열을 200으로 반환할 수 있으므로
    .json() 실패 시 진단 정보를 포함해 예외를 올림.
    """
    last = None
    headers = {"Accept":"application/json"}
    for _ in range(retry):
        r = requests.get(url, params=params, headers=headers, timeout=30)
        try:
            data = r.json()
            # KOSIS는 오류도 200으로 내려줄 수 있음
            if isinstance(data, dict) and data.get("errMsg"):
                last = r
            else:
                return data
        except Exception:
            last = r
        time.sleep(sleep)
    if last is None:
        raise RuntimeError("KOSIS 응답 실패(연결 불가).")
    ct = last.headers.get("Content-Type")
    preview = last.text[:200].replace("\n"," ")
    raise RuntimeError(f"KOSIS JSON 파싱 실패: status={last.status_code}, "
                       f"Content-Type={ct}, preview={preview}, params={params}")

def get_range_all_items(org:str, tbl:str, prdSe:str, y0:int, y1:int) -> pd.DataFrame:
    """
    연도 구간 × 전지역(ALL) × 전항목(ALL) 조회 → 표준 컬럼만 반환
    필드: region_name, period, value, (있으면 ITM_NM/ITM_ID)
    """
    params = {
        "method":"getList","apiKey":API_KEY,"format":"json","jsonVD":"Y",
        "orgId":org,"tblId":tbl,"prdSe":prdSe,
        "startPrdDe":str(y0),"endPrdDe":str(y1),
        "objL1":"ALL","itmId":"ALL",
        # outputFields 생략(표마다 상이; 전체 받아 안전 파싱)
    }
    data = _get_json(BASE_PARAM, params)
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # 필수 컬럼 존재 확인
    if "DT" not in df.columns or "PRD_DE" not in df.columns:
        raise RuntimeError("KOSIS 응답에 DT 또는 PRD_DE 필드가 없습니다.")
    # 지역명 컬럼(C1_NM..)
    region_col = next((c for c in ["C1_NM","C2_NM","C3_NM","C4_NM","C5_NM","C6_NM","C7_NM","C8_NM"] if c in df.columns), None)
    if not region_col:
        raise RuntimeError("지역명 컬럼(C1_NM..C8_NM)을 찾지 못했습니다.")

    # 수치/연도형 변환 및 정리
    df["DT"] = pd.to_numeric(df["DT"], errors="coerce")
    df["PRD_DE"] = pd.to_numeric(df["PRD_DE"], errors="coerce")
    df = df.rename(columns={region_col:"region_name","PRD_DE":"period","DT":"value"})
    df = df.dropna(subset=["region_name","period","value"])
    # 합계행 제거
    df = df[~df["region_name"].isin({"전국","합계","계","전체"})]

    keep = ["region_name","period","value"]
    if "ITM_NM" in df.columns: keep.append("ITM_NM")
    if "ITM_ID" in df.columns: keep.append("ITM_ID")
    return df[keep]

def series_by_keyword(org:str, tbl:str, keyword:str, prdSe:str, y0:int, y1:int) -> pd.DataFrame:
    """
    ITM_NM이 있으면 contains(keyword)로 필터 → 지역×연도 평균
    ITM_NM이 없으면 ITM_ID 평균(있으면 1차) → 지역×연도 평균 (폴백)
    반환: region_name, period(int), value
    """
    raw = get_range_all_items(org, tbl, prdSe, y0, y1)
    if raw.empty:
        return pd.DataFrame(columns=["region_name","period","value"])

    # 지역명 표준화
    raw["region_name"] = raw["region_name"].astype(str).apply(normalize_region_kor)

    if "ITM_NM" in raw.columns and raw["ITM_NM"].notna().any():
        sub = raw[raw["ITM_NM"].astype(str).str.contains(keyword, na=False)]
        if sub.empty:
            cand = ", ".join(sorted(map(str, raw["ITM_NM"].dropna().unique()))[:10])
            raise RuntimeError(f"'{keyword}' 항목을 찾지 못했습니다(표 {tbl}). 가능한 예: {cand}")
        g = sub.groupby(["region_name","period"], as_index=False)["value"].mean()
    else:
        grp = ["region_name","period"] + (["ITM_ID"] if "ITM_ID" in raw.columns else [])
        tmp = raw.groupby(grp, as_index=False)["value"].mean()
        if "ITM_ID" in tmp.columns:
            tmp = tmp.groupby(["region_name","period"], as_index=False)["value"].mean()
        g = tmp

    g["period"] = g["period"].astype(int)
    return g[["region_name","period","value"]]

# ---------- BDS 계산 ----------
def compute_bds_one_year(df_year: pd.DataFrame,
                         cols: List[str],
                         weight_mode: str = "pca",
                         custom: Optional[Dict[str,float]] = None) -> Tuple[pd.DataFrame, Dict[str,float]]:
    """
    df_year: columns=[region_name, period] + cols
    - elderly_rate는 부담방향이므로 부호 반전 후 표준화
    - weight_mode: "pca" | "equal" | "custom"
    """
    X = df_year[cols].astype(float).copy()

    X_adj = X.copy()
    if "elderly_rate" in X_adj.columns:
        X_adj["elderly_rate"] = -X_adj["elderly_rate"]

    Z = StandardScaler().fit_transform(X_adj.values)

    if weight_mode == "equal":
        w = np.full(len(cols), 1/len(cols), dtype=float)
    elif weight_mode == "custom":
        if not custom:
            raise ValueError("custom 가중치 dict가 필요합니다.")
        w = np.array([custom[c] for c in cols], dtype=float)
        s = w.sum()
        if s == 0:
            raise ValueError("custom 가중치의 합이 0입니다.")
        w = w / s
    else:  # "pca"
        p = PCA(n_components=1, random_state=42).fit(Z)
        load = np.abs(p.components_[0])
        w = load / load.sum()

    df_out = df_year.copy()
    df_out["BDS"] = (Z * w).sum(axis=1)
    weights = {c: float(w[i]) for i, c in enumerate(cols)}
    return df_out, weights

# ---------- 지역발전지수 데이터 로드 ----------
def load_navis_data(file_path: str = "navis_data/1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx") -> Dict[str, pd.DataFrame]:
    """
    NAVIS 지역발전지수 데이터를 로드합니다.
    """
    try:
        # Excel 파일에서 시트별로 데이터 로드
        sheets = {
            'total': 'I지역발전지수(총합)',
            'economy': 'I지역발전지수(경제력)', 
            'living': 'I지역발전지수(주민생활력)'
        }
        
        data = {}
        for key, sheet_name in sheets.items():
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            data[key] = df
            
        return data
    except Exception as e:
        print(f"NAVIS 데이터 로드 실패: {e}")
        return {}

# ---------- 지역 간 균형점수 차이 시뮬레이션 ----------
def simulate_regional_balance_improvement(bds_data: pd.DataFrame, 
                                        target_regions: List[str] = None,
                                        improvement_rate: float = 0.1,
                                        simulation_years: int = 5) -> pd.DataFrame:
    """
    지역 간 균형점수 차이를 해결하기 위한 시뮬레이션
    
    Args:
        bds_data: BDS 데이터 (region_name, period, BDS, ...)
        target_regions: 개선 대상 지역 리스트 (None이면 하위 지역들)
        improvement_rate: 연간 개선률 (0.1 = 10%)
        simulation_years: 시뮬레이션 연도 수
    
    Returns:
        시뮬레이션 결과 데이터프레임
    """
    # 최신 연도 데이터 추출
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year].copy()
    
    # 개선 대상 지역 선정 (기본값: 하위 5개 지역)
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
                # 연도별 누적 개선 효과
                years_passed = year - latest_year
                improvement_factor = (1 + improvement_rate) ** years_passed
                year_data.loc[mask, 'BDS'] *= improvement_factor
        
        simulation_data.append(year_data)
    
    # 시뮬레이션 결과와 기존 데이터 결합
    result = pd.concat([bds_data, pd.concat(simulation_data, ignore_index=True)], 
                      ignore_index=True)
    
    return result

def analyze_regional_balance(bds_data: pd.DataFrame) -> Dict:
    """
    지역 간 균형성 분석
    
    Returns:
        균형성 지표들을 포함한 딕셔너리
    """
    latest_year = bds_data['period'].max()
    latest_data = bds_data[bds_data['period'] == latest_year]
    
    bds_values = latest_data['BDS'].values
    
    analysis = {
        'year': latest_year,
        'mean': np.mean(bds_values),
        'std': np.std(bds_values),
        'cv': np.std(bds_values) / np.mean(bds_values),  # 변동계수
        'gini': calculate_gini_coefficient(bds_values),
        'max_min_ratio': np.max(bds_values) / np.min(bds_values),
        'top_bottom_ratio': np.percentile(bds_values, 80) / np.percentile(bds_values, 20)
    }
    
    return analysis

def calculate_gini_coefficient(values: np.ndarray) -> float:
    """지니계수 계산"""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

# ---------- 지역발전지수와의 유사성 검증 ----------
def compare_with_navis_index(bds_data: pd.DataFrame, 
                           navis_data: Dict[str, pd.DataFrame],
                           target_year: int = 2019) -> Dict:
    """
    BDS와 NAVIS 지역발전지수의 유사성을 검증합니다.
    
    Args:
        bds_data: BDS 데이터
        navis_data: NAVIS 지역발전지수 데이터
        target_year: 비교 대상 연도
    
    Returns:
        유사성 분석 결과
    """
    # BDS 데이터에서 해당 연도 추출
    bds_year = bds_data[bds_data['period'] == target_year].copy()
    
    if bds_year.empty:
        return {"error": f"BDS 데이터에 {target_year}년 데이터가 없습니다."}
    
    # NAVIS 데이터에서 해당 연도 추출 (시트별로)
    navis_year_data = {}
    for key, df in navis_data.items():
        # NAVIS 데이터 구조에 따라 연도 컬럼 찾기
        year_col = None
        for col in df.columns:
            if str(target_year) in str(df[col].iloc[0]):
                year_col = col
                break
        
        if year_col:
            navis_year_data[key] = df[['지역명', year_col]].copy()
            navis_year_data[key].columns = ['region_name', 'navis_index']
    
    # 지역명 매칭 및 상관관계 분석
    results = {}
    
    for navis_type, navis_df in navis_year_data.items():
        # 지역명 정규화
        navis_df['region_name'] = navis_df['region_name'].apply(normalize_region_kor)
        bds_temp = bds_year.copy()
        bds_temp['region_name'] = bds_temp['region_name'].apply(normalize_region_kor)
        
        # 공통 지역 찾기
        common_regions = set(bds_temp['region_name']) & set(navis_df['region_name'])
        
        if len(common_regions) < 3:
            results[navis_type] = {"error": "공통 지역이 부족합니다."}
            continue
        
        # 데이터 병합
        merged = bds_temp.merge(navis_df, on='region_name', how='inner')
        
        if len(merged) < 3:
            results[navis_type] = {"error": "병합 후 데이터가 부족합니다."}
            continue
        
        # 상관관계 분석
        correlation = merged['BDS'].corr(merged['navis_index'])
        
        # 순위 상관관계 (스피어만)
        rank_correlation = merged['BDS'].corr(merged['navis_index'], method='spearman')
        
        # 순위 일치도 계산
        bds_rank = merged['BDS'].rank(ascending=False)
        navis_rank = merged['navis_index'].rank(ascending=False)
        rank_agreement = 1 - (np.abs(bds_rank - navis_rank) / len(merged)).mean()
        
        results[navis_type] = {
            'correlation': correlation,
            'rank_correlation': rank_correlation,
            'rank_agreement': rank_agreement,
            'common_regions': len(merged),
            'regions': merged['region_name'].tolist()
        }
    
    return results

# ---------- 시각화 함수들 ----------
def plot_regional_balance_simulation(simulation_data: pd.DataFrame, 
                                   target_regions: List[str] = None):
    """지역 균형 개선 시뮬레이션 시각화"""
    fig = px.line(simulation_data, x='period', y='BDS', color='region_name',
                  title='지역 균형 개선 시뮬레이션',
                  labels={'period': '연도', 'BDS': 'BDS 점수', 'region_name': '지역'})
    
    # 대상 지역 강조
    if target_regions:
        for region in target_regions:
            region_data = simulation_data[simulation_data['region_name'] == region]
            fig.add_trace(go.Scatter(
                x=region_data['period'], y=region_data['BDS'],
                mode='lines+markers', name=f'{region} (개선대상)',
                line=dict(width=3), marker=dict(size=8)
            ))
    
    fig.update_layout(height=600)
    return fig

def plot_correlation_with_navis(bds_data: pd.DataFrame, 
                               navis_data: Dict[str, pd.DataFrame],
                               target_year: int = 2019):
    """BDS와 NAVIS 지수의 상관관계 시각화"""
    # 데이터 준비
    bds_year = bds_data[bds_data['period'] == target_year].copy()
    
    navis_year_data = {}
    for key, df in navis_data.items():
        year_col = None
        for col in df.columns:
            if str(target_year) in str(df[col].iloc[0]):
                year_col = col
                break
        
        if year_col:
            navis_year_data[key] = df[['지역명', year_col]].copy()
            navis_year_data[key].columns = ['region_name', 'navis_index']
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=1, cols=len(navis_year_data),
        subplot_titles=[f'BDS vs {key}' for key in navis_year_data.keys()],
        specs=[[{"secondary_y": False}] * len(navis_year_data)]
    )
    
    colors = ['blue', 'red', 'green']
    
    for i, (navis_type, navis_df) in enumerate(navis_year_data.items(), 1):
        # 지역명 정규화
        navis_df['region_name'] = navis_df['region_name'].apply(normalize_region_kor)
        bds_temp = bds_year.copy()
        bds_temp['region_name'] = bds_temp['region_name'].apply(normalize_region_kor)
        
        # 데이터 병합
        merged = bds_temp.merge(navis_df, on='region_name', how='inner')
        
        if len(merged) >= 3:
            # 산점도
            fig.add_trace(
                go.Scatter(
                    x=merged['BDS'], y=merged['navis_index'],
                    mode='markers+text',
                    text=merged['region_name'],
                    textposition="top center",
                    name=navis_type,
                    marker=dict(color=colors[i-1], size=10),
                    showlegend=False
                ),
                row=1, col=i
            )
            
            # 상관계수 표시
            corr = merged['BDS'].corr(merged['navis_index'])
            fig.add_annotation(
                x=0.5, y=0.95,
                xref=f'x{i}', yref=f'y{i}',
                text=f'상관계수: {corr:.3f}',
                showarrow=False,
                font=dict(size=12)
            )
    
    fig.update_layout(
        title=f'BDS와 NAVIS 지역발전지수 상관관계 ({target_year}년)',
        height=500,
        showlegend=False
    )
    
    return fig

# ---------- 메인: 최근 10년 시계열 + 지도 ----------
def build_timeseries_and_map(weight_mode="pca",
                             custom_weights: Optional[Dict[str,float]] = None,
                             min_year: int = 2000,
                             geojson_path: str = "./skorea-provinces-2018-geo.json",
                             geojson_name_field: Optional[str] = None,
                             out_csv: str = "bds_timeseries.csv",
                             out_html: str = "bds_choropleth.html"):
    """
    1) 각 표에서 넓은 구간(min_year~2100)을 한 번 호출하여 이용 가능한 연도 목록 확보
    2) 4지표 연도 교집합의 최댓값을 end_year로 채택 → 최근 10년(end_year-9 ~ end_year)
    3) 연도별로 4표 inner join → BDS 산출
    4) CSV/HTML 저장
    """
    # 1) 넓은 구간에서 지표별 연도 확보
    avail = {}
    wide_y0, wide_y1 = min_year, 2100
    for key, spec in TABLES.items():
        dfw = series_by_keyword(spec["org"], spec["tbl"], spec["keyword"], spec["prdSe"], wide_y0, wide_y1)
        years = sorted(dfw["period"].unique().tolist())
        if not years:
            raise RuntimeError(f"{key} 표에서 사용 가능한 연도가 없습니다. 표ID/항목/주기를 확인하세요.")
        avail[key] = years

    # 2) 4지표 연도 교집합 → end_year 및 최근 10년
    inter = set(avail["pc_grdp"])
    for k in ["pop_growth","elderly_rate","fiscal_indep"]:
        inter &= set(avail[k])
    if not inter:
        raise RuntimeError("4개 지표의 연도 교집합이 없습니다. 표/항목/주기를 확인하세요.")
    end_year = max(inter)
    years = [y for y in range(end_year-9, end_year+1) if y in inter]
    if not years:
        raise RuntimeError("최근 10년을 구성할 연도가 부족합니다.")
    print(f"[INFO] 사용 연도: {years[0]}–{years[-1]} (end_year={end_year})")

    # 3) 연도별 수집·병합
    frames = []
    for y in years:
        parts = []
        for col, spec in TABLES.items():
            df = series_by_keyword(spec["org"], spec["tbl"], spec["keyword"], spec["prdSe"], y, y)
            df = df.rename(columns={"value": col})
            parts.append(df[["region_name","period",col]])
        one = parts[0]
        for p in parts[1:]:
            one = one.merge(p, on=["region_name","period"], how="inner")
        if not one.empty:
            frames.append(one)

    if not frames:
        raise RuntimeError("연도별 병합 결과가 비었습니다.")
    base_ts = pd.concat(frames, ignore_index=True)
    base_ts["region_name"] = base_ts["region_name"].apply(normalize_region_kor)

    # 4) 연도별 BDS
    cols = ["pc_grdp","pop_growth","elderly_rate","fiscal_indep"]
    bds_frames, weights_by_year = [], []
    for y, dfy in base_ts.groupby("period"):
        scored, w = compute_bds_one_year(dfy, cols, weight_mode, custom_weights)
        bds_frames.append(scored)
        w["period"] = int(y)
        weights_by_year.append(w)

    bds_ts = pd.concat(bds_frames, ignore_index=True)
    bds_ts = bds_ts.sort_values(["period","BDS"], ascending=[True, False]).reset_index(drop=True)

    # 5) 저장
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bds_ts.to_csv(OUTDIR/out_csv, index=False, encoding="utf-8-sig")
    with open(OUTDIR/"weights_by_year.json","w",encoding="utf-8") as f:
        json.dump({"weight_mode":weight_mode,
                   "weights_by_year":weights_by_year,
                   "years":years,
                   "end_year":int(end_year)}, f, ensure_ascii=False, indent=2)

    # 6) 지도(연도 슬라이더 코로플레스) — 한글/영문 자동 매칭 패치
    with open(geojson_path,"r",encoding="utf-8") as f:
        gj = json.load(f)

    # 시도명 필드 자동판별: 'CTP_KOR_NM'(한글) 우선, 없으면 'name'
    if geojson_name_field is None:
        sample_props = gj["features"][0]["properties"]
        if "CTP_KOR_NM" in sample_props:
            geojson_name_field = "CTP_KOR_NM"
        elif "name" in sample_props:
            geojson_name_field = "name"
        else:
            raise KeyError(f"GeoJSON에서 시도명 필드를 찾지 못했습니다. keys={list(sample_props.keys())}")

    # 'name'이 한글인지/영문인지 감지
    def _has_hangul(s: str) -> bool:
        s = str(s)
        return any("\uac00" <= ch <= "\ud7a3" for ch in s)

    name_samples = [feat["properties"][geojson_name_field] for feat in gj["features"]]
    uses_korean_names = any(_has_hangul(v) for v in name_samples)

    df_map = bds_ts.copy()

    if geojson_name_field == "CTP_KOR_NM":
        # 한글 시도명
        df_map["loc_key"] = df_map["region_name"].astype(str)
        featureidkey = "properties.CTP_KOR_NM"

    elif geojson_name_field == "name":
        if uses_korean_names:
            # name이 한글 → 영어 매핑 금지, 그대로 매칭
            df_map["loc_key"] = df_map["region_name"].astype(str)
        else:
            # name이 영문 → 한→영 매핑
            df_map["loc_key"] = df_map["region_name"].map(lambda x: KOR2ENG.get(x, x))
        featureidkey = "properties.name"

    else:
        # 기타 필드: region_name 그대로 매칭
        df_map["loc_key"] = df_map["region_name"].astype(str)
        featureidkey = f"properties.{geojson_name_field}"

    # 매칭 검증: GeoJSON에 없는 지역명 출력
    gj_name_set = {feat["properties"][geojson_name_field] for feat in gj["features"]}
    missing = sorted(set(df_map["loc_key"]) - gj_name_set)
    if missing:
        print("⚠ 매칭 실패 지역명(GeoJSON에 없음):", missing)

    fig = px.choropleth(
        df_map,
        geojson=gj,
        featureidkey=featureidkey,
        locations="loc_key",
        color="BDS",
        animation_frame="period",
        color_continuous_scale="Viridis",
        hover_name="region_name",
        hover_data={"pc_grdp":":.0f","pop_growth":":.2f","elderly_rate":":.2f",
                    "fiscal_indep":":.1f","loc_key":False}
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(title=f"BDS 최근 10년(가중치: {weight_mode})",
                      margin=dict(l=0, r=0, t=45, b=0))
    out_html_path = OUTDIR/out_html
    fig.write_html(str(out_html_path), include_plotlyjs="cdn")

    print(f"[완료] 시계열 CSV: {OUTDIR/out_csv}")
    print(f"[완료] 지도 HTML: {out_html_path}")
    
    return bds_ts

# ---- 실행 예시 ----
if __name__ == "__main__":
    # 준비한 시도 경계 GeoJSON 경로 지정
    # (예: 업로드한 파일명)
    GEOJSON_PATH = "./skorea-provinces-2018-geo.json"

    # BDS 계산 및 시각화
    bds_data = build_timeseries_and_map(
        weight_mode="pca",          # "equal" 또는 "custom"도 가능
        custom_weights=None,        # 예: {"pc_grdp":0.35,"pop_growth":0.25,"elderly_rate":0.15,"fiscal_indep":0.25}
        min_year=2000,
        geojson_path=GEOJSON_PATH,
        geojson_name_field=None,    # 수동지정하려면 "name" 또는 "CTP_KOR_NM"
        out_csv="bds_timeseries.csv",
        out_html="bds_choropleth.html"
    )
    
    # 지역 균형성 분석
    balance_analysis = analyze_regional_balance(bds_data)
    print("\n=== 지역 균형성 분석 ===")
    for key, value in balance_analysis.items():
        print(f"{key}: {value:.4f}")
    
    # NAVIS 데이터 로드 및 비교
    navis_data = load_navis_data()
    if navis_data:
        comparison_results = compare_with_navis_index(bds_data, navis_data)
        print("\n=== NAVIS 지역발전지수와의 유사성 ===")
        for navis_type, result in comparison_results.items():
            if "error" not in result:
                print(f"\n{navis_type}:")
                print(f"  상관계수: {result['correlation']:.3f}")
                print(f"  순위상관계수: {result['rank_correlation']:.3f}")
                print(f"  순위일치도: {result['rank_agreement']:.3f}")
                print(f"  공통지역 수: {result['common_regions']}")
            else:
                print(f"\n{navis_type}: {result['error']}")
    
    # 지역 균형 개선 시뮬레이션
    simulation_data = simulate_regional_balance_improvement(bds_data)
    print(f"\n=== 지역 균형 개선 시뮬레이션 완료 ===")
    print(f"시뮬레이션 데이터 포인트: {len(simulation_data)}")
    
    # 시각화 저장
    if navis_data:
        corr_fig = plot_correlation_with_navis(bds_data, navis_data)
        corr_fig.write_html(str(OUTDIR / "bds_navis_correlation.html"))
        print("[완료] NAVIS 상관관계 HTML: outputs_timeseries/bds_navis_correlation.html")
    
    sim_fig = plot_regional_balance_simulation(simulation_data)
    sim_fig.write_html(str(OUTDIR / "regional_balance_simulation.html"))
    print("[완료] 지역 균형 시뮬레이션 HTML: outputs_timeseries/regional_balance_simulation.html") 