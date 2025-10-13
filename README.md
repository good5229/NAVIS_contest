# NABIS 공모전 - 지역균형발전지수(BDS) 정책 시뮬레이터

## 📊 프로젝트 개요

국토교통부 NABIS(지역균형발전지수) 공모전을 위한 **BDS(Balanced Development Score) 정책 시뮬레이터**입니다. EU RCI 방법론을 적용한 한국형 지역균형발전지수를 활용하여 정책 시나리오별 효과를 시뮬레이션하고 시각화하는 통합 분석 플랫폼입니다.

## 🌐 라이브 데모

**GitHub Pages**: https://good5229.github.io/NAVIS_contest/

### 주요 페이지:
- **메인 대시보드**: https://good5229.github.io/NAVIS_contest/bds_enhanced_dashboard.html
- **정책 시뮬레이터**: https://good5229.github.io/NAVIS_contest/bds_simulator.html

## 📁 프로젝트 구조

```
NABIS_contest/
├── bds_enhanced_dashboard.html   # BDS 대시보드 (메인)
├── bds_simulator.html            # BDS 정책 시뮬레이터
├── fiscal_autonomy_dashboard.html # 재정자립도 대시보드
├── fiscal_simulator.html         # 재정정책 시뮬레이터
├── 울산_조선업_BDS_연관성_분석.html # 울산 조선업 선행성 분석
├── data/                        # 데이터 폴더
│   ├── bds/                     # BDS 관련 데이터
│   ├── ecos/                    # 한국은행 ECOS 데이터
│   ├── kosis/                   # 통계청 KOSIS 데이터
│   └── nabis/                   # NABIS 객관지표 데이터
├── scripts/                     # 분석 스크립트
├── docs/                        # 분석 보고서
└── screenshots/                 # 스크린샷 이미지
```

## 🎯 주요 기능

### 1. BDS 대시보드 (메인)
- **분포 분석**: 히스토그램과 박스플롯을 통한 BDS 분포 시각화
- **상관관계 분석**: NABIS와 BDS 간의 상관관계 분석
- **선행성 분석**: 그레인저 인과관계 검정 결과
- **지역별 변화**: 지역별 BDS 변화 추이

### 2. BDS 정책 시뮬레이터
- **9가지 정책 시나리오**: 균형발전, 경제중심, 사회중심, 혁신중심, 환경중심, 문화예술중심, 안전중심, 주거중심, R&D중심
- **지역별 효과 분석**: 17개 시도별 정책 효과 시뮬레이션
- **정책 카테고리별 효과**: 10개 정책 영역별 효과 분석
- **인터랙티브 테이블**: 정렬 가능한 지역별 상세 결과

### 3. 재정자립도 분석
- **재정자립도 대시보드**: 지역별 재정자립도 현황 분석
- **재정정책 시뮬레이터**: 재정정책 효과 시뮬레이션
- **낮은 재정자립도 지역**: 특화 분석 및 정책 제안

### 4. 울산 조선업 선행성 분석
- **제조업 생산액 vs BDS**: 울산 지역 제조업과 BDS의 선행성 분석
- **NABIS 비교**: BDS와 NABIS의 선행성 비교
- **그레인저 인과관계**: 통계적 선행성 검증

## 🔬 학술적 근거

- **EU RCI 방법론**: 유럽연합 지역경쟁력지수 가중치 체계 적용
- **System Dynamics**: Sterman(2000), Forrester(1961) 이론 기반
- **OECD & World Bank**: 정책 효과 계수 산정
- **Solow Growth Model**: 한계효용 체감 이론 적용
- **그레인저 인과관계**: Granger(1969) 선행성 검정 방법론

## 📊 데이터 소스

- **ECOS**: 한국은행 경제통계시스템 (제조업 생산지수, GRDP 등)
- **KOSIS**: 통계청 국가통계포털 (재정자립도, 인구, 고용 등)
- **NABIS**: 국토교통부 지역균형발전지수 (객관지표 47개 파일)
- **실제 데이터**: 2015-2023년 시계열 데이터 활용

## 🛠️ 기술 스택

- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Bootstrap 5
- **Charts**: Plotly.js
- **Icons**: Font Awesome 6
- **Deployment**: GitHub Pages

## 🏆 NABIS 공모전 참가작

이 프로젝트는 **국토교통부 NABIS(지역균형발전지수) 공모전**을 위해 제작되었습니다.

### 주요 성과:
- **실제 데이터 기반**: 가상 데이터 없이 실제 공공데이터만 활용
- **학술적 근거**: EU RCI, System Dynamics 등 국제 표준 방법론 적용
- **실용적 도구**: 정책 입안자와 연구자가 활용할 수 있는 시뮬레이터
- **시각화**: 직관적인 대시보드와 인터랙티브 차트

## ⚠️ 주의사항

- 이 데모는 **학술 연구 목적**으로 제작되었습니다
- 실제 정책 결정에 사용하기 전 추가 검증이 필요합니다
- 모든 데이터는 **실제 공공데이터**를 기반으로 합니다

## 📞 문의

프로젝트 관련 문의사항이 있으시면 GitHub Issues를 통해 연락해주세요.

---

**© 2025 NABIS 공모전 참가작. 국토교통부 지역균형발전지수 공모전을 위한 연구 프로젝트입니다.**
