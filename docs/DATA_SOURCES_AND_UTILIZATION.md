# 데이터 소스 및 활용 현황

## 데이터 수집 방식별 분류

### 1. API 수집 데이터

#### ECOS (한국은행 경제통계시스템)
- **수집 방식**: Python 스크립트를 통한 API 호출
- **데이터 기간**: 2015-2023
- **파일 위치**: `data/ecos/`
- **주요 지표**:
  - 지역내총생산(GRDP)
  - 제조업 생산지수
  - 서비스업 생산지수
  - 고용지수
- **파일 수**: 10개 (8개 CSV, 2개 JSON)
- **용도**: BDS 계산의 경제 지표 기반 데이터

#### KOSIS (통계청 국가통계포털)
- **수집 방식**: Python 스크립트를 통한 API 호출
- **데이터 기간**: 2015-2023
- **파일 위치**: `data/kosis/`
- **주요 지표**:
  - 재정자립도
  - 인구 통계
  - 주택 보급률
  - 교육 통계
- **파일 수**: 10개 (5개 XLSX, 2개 JSON, 2개 XML, 1개 CSV)
- **용도**: 재정자립도 분석, 사회 지표 데이터

### 2. 파일 로드 데이터

#### NABIS (국토교통부 국가균형발전종합정보시스템)
- **수집 방식**: Excel 파일 직접 다운로드
- **데이터 기간**: 2015-2023 (지표별 상이)
- **파일 위치**: `data/nabis/`
- **주요 지표**:
  - 지역발전지수(RDI)
  - 객관지표 (47개 Excel 파일)
  - 시계열 자료
- **파일 수**: 4개 (3개 JSON, 1개 XLSX)
- **용도**: 정책 시뮬레이터, RDI 기반 분석

#### BDS (지역균형발전지수)
- **수집 방식**: 계산된 결과 데이터
- **데이터 기간**: 2015-2023
- **파일 위치**: `data/bds/`
- **주요 지표**:
  - BDS 점수
  - NABIS 비교 데이터
  - 시뮬레이션 결과
- **파일 수**: 8개 (5개 JSON, 3개 CSV)
- **용도**: 메인 대시보드, 시각화

## 데이터별 상세 정보

### 1. ECOS 데이터
```
data/ecos/
├── grdp_data.csv (17행 × 9열) - 지역별 GRDP 데이터
├── manufacturing_index.csv (17행 × 9열) - 제조업 생산지수
├── service_index.csv (17행 × 9열) - 서비스업 생산지수
├── employment_index.csv (17행 × 9열) - 고용지수
├── ecos_metadata.json - 메타데이터
└── ecos_collection_log.json - 수집 로그
```

### 2. KOSIS 데이터
```
data/kosis/
├── fiscal_autonomy_data.xlsx (17행 × 9열) - 재정자립도
├── population_data.xlsx (17행 × 9열) - 인구 통계
├── housing_data.xlsx (17행 × 9열) - 주택 보급률
├── education_data.xlsx (17행 × 9열) - 교육 통계
├── kosis_metadata.json - 메타데이터
└── kosis_collection_log.json - 수집 로그
```

### 3. NABIS 데이터
```
data/nabis/
├── rdi_simulation_data.json - RDI 시뮬레이션 데이터
├── category_mapping.json - 지표 카테고리 매핑
├── policy_rdi_mapping.json - 정책-RDI 매핑
└── 1_2. 시계열자료(사이트게재)_지역발전지수_2021년.xlsx - 원본 데이터
```

### 4. BDS 데이터
```
data/bds/
├── bds_baseline.json - 대시보드/시뮬레이터 기준 BDS (최신 연도)
├── bds_yearly_baselines.json - 연도별 베이스라인 집계
├── bds_full_timeseries_results.csv - 전체 시계열 산출
├── bds_weighted_results.json - 가중치 적용 결과 (요약)
├── bds_weighted_results.csv - 가중치 적용 결과 (상세)
├── bds_detailed_2023.csv - 2023년 상세 산출
└── real_granger_analysis_results.json - 선행성 분석 결과
```

## 데이터 활용 현황

### 1. BDS 대시보드
- **주요 데이터**: BDS 시계열, 지역별 데이터
- **시각화**: Plotly.js 차트, Leaflet 지도
- **기능**: 지역별 BDS 추이, 상관관계 분석, 선행성 분석

### 2. 정책 시뮬레이터
- **주요 데이터**: NABIS RDI 데이터, 정책 매핑
- **시각화**: 정책 효과 차트, 지역별 개선 효과
- **기능**: 9개 정책 시나리오 시뮬레이션

### 3. 재정자립도 분석
- **주요 데이터**: KOSIS 재정자립도 데이터
- **시각화**: 재정자립도 추이, 지역별 비교
- **기능**: 낮은 재정자립도 지역 분석

### 4. 울산 조선업 선행성 분석
- **주요 데이터**: ECOS 제조업 생산지수, BDS/NABIS 데이터
- **시각화**: 선행성 분석 차트
- **기능**: 제조업 생산과 BDS/NABIS 상관관계 분석

## 데이터 품질 관리

### 1. 데이터 검증
```python
def validate_data(data, required_fields):
    """데이터 유효성 검증"""
    for field in required_fields:
        if field not in data:
            raise ValueError(f"필수 필드 누락: {field}")
    
    # 데이터 타입 검증
    if not isinstance(data, dict):
        raise TypeError("데이터는 딕셔너리 형식이어야 합니다")
    
    return True
```

### 2. 데이터 정제
```python
def clean_data(raw_data):
    """데이터 정제 및 전처리"""
    cleaned_data = {}
    
    for key, value in raw_data.items():
        # 결측값 처리
        if value is None or value == '':
            cleaned_data[key] = 0
        else:
            cleaned_data[key] = float(value)
    
    return cleaned_data
```

### 3. 데이터 일관성 확인
```python
def check_data_consistency(data1, data2):
    """데이터 일관성 확인"""
    common_keys = set(data1.keys()) & set(data2.keys())
    
    for key in common_keys:
        if abs(data1[key] - data2[key]) > 0.01:
            print(f"데이터 불일치 발견: {key}")
            return False
    
    return True
```

## 데이터 보안 및 개인정보 보호

### 1. 데이터 익명화
- 모든 개인 식별 정보 제거
- 지역명만 사용, 개인 데이터 없음
- 집계된 통계 데이터만 활용

### 2. 데이터 접근 제어
- 공개 데이터만 사용
- 민감한 정보 포함 데이터 제외
- 정부 공식 통계 데이터 우선 사용

### 3. 데이터 백업
- Git을 통한 버전 관리
- 정기적인 데이터 백업
- 데이터 손실 방지 체계

## 데이터 업데이트 정책

### 1. 정기 업데이트
- **ECOS 데이터**: 월 1회
- **KOSIS 데이터**: 분기 1회
- **NABIS 데이터**: 연 1회

### 2. 업데이트 프로세스
1. 새 데이터 수집
2. 데이터 검증 및 정제
3. 기존 데이터와 비교
4. 변경사항 문서화
5. 시스템 업데이트

### 3. 버전 관리
```json
{
    "data_version": "2024.01",
    "last_updated": "2024-01-15",
    "update_frequency": "monthly",
    "data_sources": {
        "ecos": "2024-01-10",
        "kosis": "2024-01-05",
        "nabis": "2024-01-01"
    }
}
```

## 데이터 활용 가이드

### 1. 새로운 데이터 추가 시
1. 데이터 소스 확인 및 검증
2. 데이터 형식 표준화
3. 메타데이터 작성
4. 테스트 및 검증
5. 문서화

### 2. 데이터 수정 시
1. 백업 생성
2. 변경사항 문서화
3. 영향도 분석
4. 테스트 및 검증
5. 배포

### 3. 데이터 삭제 시
1. 사용 현황 확인
2. 의존성 분석
3. 백업 확인
4. 단계적 제거
5. 문서 업데이트

## 문제 해결 가이드

### 1. 데이터 로딩 실패
- 네트워크 연결 확인
- 파일 경로 확인
- 데이터 형식 검증
- 권한 확인

### 2. 데이터 불일치
- 데이터 소스 확인
- 수집 시점 확인
- 계산 로직 검증
- 버전 관리 확인

### 3. 성능 문제
- 데이터 크기 확인
- 캐싱 전략 검토
- 로딩 방식 최적화
- 메모리 사용량 모니터링
