# BDS 정책 시뮬레이터 - 웹 데모

## 📊 프로젝트 개요

BDS(Balanced Development Score) 정책 시뮬레이터는 EU RCI 방법론을 적용한 한국형 지역균형발전지수를 활용하여 정책 시나리오별 효과를 시뮬레이션하고 시각화하는 통합 분석 플랫폼입니다.

## 🚀 GitHub Pages 배포

이 폴더는 GitHub Pages를 통해 웹 데모로 배포됩니다.

### 배포 방법:
1. GitHub 저장소의 Settings → Pages
2. Source를 "Deploy from a branch"로 설정
3. Branch를 "main"으로, Folder를 "/web"으로 설정
4. Save 후 자동 배포 완료

## 📁 파일 구조

```
web/
├── index.html                    # 메인 페이지
├── bds_enhanced_dashboard.html   # BDS 대시보드
├── bds_simulator.html           # BDS 시뮬레이터
├── screenshots/                 # 스크린샷 이미지
│   ├── balance_index_trend.png
│   ├── bds_map.png
│   ├── bds_trend_chart.png
│   └── correlation_analysis.png
└── README.md                    # 이 파일
```

## 🎯 주요 기능

### 1. BDS 대시보드
- **분포 분석**: 히스토그램과 박스플롯을 통한 BDS 분포 시각화
- **상관관계 분석**: NABIS와 BDS 간의 상관관계 분석
- **선행성 분석**: 그레인저 인과관계 검정 결과
- **지역별 변화**: 지역별 BDS 변화 추이

### 2. BDS 시뮬레이터
- **9가지 정책 시나리오**: 균형발전, 경제중심, 사회중심, 혁신중심, 환경중심, 문화예술중심, 안전중심, 주거중심, R&D중심
- **지역별 효과 분석**: 17개 시도별 정책 효과 시뮬레이션
- **정책 카테고리별 효과**: 10개 정책 영역별 효과 분석
- **인터랙티브 테이블**: 정렬 가능한 지역별 상세 결과

## 🔬 학술적 근거

- **EU RCI 방법론**: 유럽연합 지역경쟁력지수 가중치 체계 적용
- **System Dynamics**: Sterman(2000), Forrester(1961) 이론 기반
- **OECD & World Bank**: 정책 효과 계수 산정
- **Solow Growth Model**: 한계효용 체감 이론 적용

## 📊 데이터 소스

- **ECOS**: 한국은행 경제통계시스템
- **KOSIS**: 통계청 국가통계포털
- **NABIS**: 국토교통부 지역균형발전지수

## 🛠️ 기술 스택

- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Bootstrap 5
- **Charts**: Plotly.js
- **Icons**: Font Awesome 6
- **Deployment**: GitHub Pages

## ⚠️ 주의사항

- 이 데모는 **학술 연구 목적**으로 제작되었습니다
- 실제 정책 결정에 사용하기 전 추가 검증이 필요합니다
- 데이터는 2019년 기준으로 제한됩니다

## 📞 문의

프로젝트 관련 문의사항이 있으시면 GitHub Issues를 통해 연락해주세요.

---

**© 2025 BDS 정책 시뮬레이터. 학술 연구 목적의 데모 버전입니다.**
