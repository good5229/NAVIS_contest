# 배포 및 운영 가이드

## GitHub Pages 배포 설정

### 1. 저장소 설정
```bash
# 저장소 클론
git clone https://github.com/good5229/NABIS_contest.git
cd NABIS_contest

# GitHub Pages 설정 확인
# Settings > Pages > Source: Deploy from a branch
# Branch: main
# Folder: / (root)
```

### 2. 배포 파일 구조
```
/NABIS_contest/
├── index.html                    # 메인 랜딩 페이지
├── bds_enhanced_dashboard.html   # BDS 대시보드
├── bds_simulator.html           # 정책 시뮬레이터
├── data/                        # 데이터 파일들
├── scripts/                     # 스크립트 파일들
└── docs/                        # 문서 파일들
```

### 3. 배포 명령어
```bash
# 변경사항 커밋
git add .
git commit -m "Update: 시뮬레이터 개선 효과 계산 수정"

# GitHub Pages에 배포
git push origin main

# 강제 배포 (캐시 문제 해결 시)
git commit --amend --no-edit
git push --force origin main
```

## 로컬 개발 환경 설정

### 1. Python 환경 설정
```bash
# Python 3.10+ 설치 확인
python3 --version

# 가상환경 생성 (선택사항)
python3 -m venv navis_env
source navis_env/bin/activate  # macOS/Linux
# navis_env\Scripts\activate   # Windows

# 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 2. 로컬 서버 실행
```bash
# HTTP 서버 실행
python3 -m http.server 8000 --bind 127.0.0.1

# 브라우저에서 접속
# http://localhost:8000
```

### 3. 개발 도구 설정
```bash
# Git hooks 설정
chmod +x .git/hooks/pre-push

# 테스트 실행
python3 scripts/test_simulator_improvements.py
```

## 배포 전 체크리스트

### 1. 코드 품질 확인
- [ ] 모든 파일이 올바른 위치에 있는지 확인
- [ ] JavaScript 문법 오류 없음
- [ ] CSS 스타일 일관성 확인
- [ ] HTML 구조 검증

### 2. 기능 테스트
- [ ] 대시보드 로딩 정상
- [ ] 시뮬레이터 정상 작동
- [ ] 지도 시각화 정상
- [ ] 반응형 디자인 확인
- [ ] 모든 링크 정상 작동

### 3. 데이터 검증
- [ ] 모든 데이터 파일 존재
- [ ] 데이터 형식 정확성
- [ ] 시뮬레이션 결과 정확성
- [ ] 메타데이터 일관성

### 4. 성능 확인
- [ ] 페이지 로딩 속도
- [ ] 차트 렌더링 속도
- [ ] 지도 로딩 속도
- [ ] 모바일 성능

## 자동화된 테스트

### 1. Pre-push Hook
```bash
#!/bin/bash
# .git/hooks/pre-push

echo "배포 전 테스트 실행 중..."

# 시뮬레이터 개선 효과 테스트
python3 scripts/test_simulator_improvements.py
if [ $? -ne 0 ]; then
    echo "❌ 시뮬레이터 테스트 실패: 푸시가 중단됩니다."
    exit 1
fi

# JavaScript 문법 테스트
python3 scripts/js_error_test.py
if [ $? -ne 0 ]; then
    echo "❌ JavaScript 테스트 실패: 푸시가 중단됩니다."
    exit 1
fi

echo "✅ 모든 테스트 통과: 푸시를 진행합니다."
```

### 2. 테스트 스크립트
```python
# scripts/test_simulator_improvements.py
def test_simulator_improvements():
    """시뮬레이터 개선 효과 테스트"""
    expected_improvements = {
        '균형발전': 0.5,
        '경제중심': 0.4,
        '사회중심': 0.35,
        '환경중심': 0.3,
        '문화예술중심': 0.25,
        '안전중심': 0.3,
        '주거중심': 0.25,
        'R&D중심': 0.4,
        '혁신중심': 0.4
    }
    
    # bds_simulator.html 파일 검증
    with open('bds_simulator.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 잘못된 값 확인
    if '70.38%' in content:
        print("❌ 잘못된 개선 효과 값 발견: 70.38%")
        return False
    
    # 올바른 값 확인
    for scenario, expected in expected_improvements.items():
        if f'"{scenario}": {expected}' not in content:
            print(f"❌ {scenario} 시나리오의 개선 효과 값이 올바르지 않음")
            return False
    
    print("✅ 모든 시뮬레이터 개선 효과 값이 올바름")
    return True
```

## 모니터링 및 로깅

### 1. 에러 로깅
```javascript
// 브라우저 콘솔 로깅
function logError(error, context) {
    console.error(`[${new Date().toISOString()}] ${context}:`, error);
    
    // 필요시 서버로 에러 전송
    if (window.location.hostname !== 'localhost') {
        fetch('/api/log-error', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                error: error.message,
                context: context,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent
            })
        }).catch(console.error);
    }
}
```

### 2. 성능 모니터링
```javascript
// 페이지 로딩 시간 측정
window.addEventListener('load', function() {
    const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
    console.log(`페이지 로딩 시간: ${loadTime}ms`);
    
    // 느린 로딩 경고
    if (loadTime > 3000) {
        console.warn('페이지 로딩이 느립니다:', loadTime + 'ms');
    }
});
```

### 3. 사용자 행동 추적
```javascript
// 시뮬레이터 사용 통계
function trackSimulatorUsage(scenario, result) {
    const usage = {
        scenario: scenario,
        improvement: result.summary.averageImprovement,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent
    };
    
    console.log('시뮬레이터 사용:', usage);
    
    // 로컬 스토리지에 저장
    const history = JSON.parse(localStorage.getItem('simulator_history') || '[]');
    history.push(usage);
    localStorage.setItem('simulator_history', JSON.stringify(history));
}
```

## 문제 해결 가이드

### 1. GitHub Pages 배포 문제
**문제**: 로컬과 GitHub Pages 결과 불일치
**해결**:
```bash
# 1. 파일 위치 확인
ls -la bds_simulator.html
ls -la web/bds_simulator.html

# 2. 강제 배포
git commit --amend --no-edit
git push --force origin main

# 3. 브라우저 캐시 클리어
# 개발자 도구 > Network > Disable cache
```

### 2. 데이터 로딩 실패
**문제**: JSON 파일 로딩 실패
**해결**:
```javascript
// 에러 핸들링 개선
async function loadData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('데이터 로딩 실패:', error);
        // 폴백 데이터 사용
        return getFallbackData();
    }
}
```

### 3. 지도 렌더링 문제
**문제**: Leaflet 지도가 표시되지 않음
**해결**:
```javascript
// 지도 초기화 확인
function initMap() {
    if (!window.L) {
        console.error('Leaflet 라이브러리가 로드되지 않았습니다.');
        return;
    }
    
    // 지도 컨테이너 확인
    const mapContainer = document.getElementById('map-container');
    if (!mapContainer) {
        console.error('지도 컨테이너를 찾을 수 없습니다.');
        return;
    }
    
    // 지도 생성
    const map = L.map('map-container').setView([36.2, 127.6], 6);
    // ... 나머지 지도 설정
}
```

## 백업 및 복구

### 1. 데이터 백업
```bash
# 전체 프로젝트 백업
tar -czf nabis_contest_backup_$(date +%Y%m%d).tar.gz \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='*.log' \
    .

# Git 백업
git bundle create nabis_contest_$(date +%Y%m%d).bundle --all
```

### 2. 복구 절차
```bash
# 백업에서 복구
tar -xzf nabis_contest_backup_20240115.tar.gz

# Git 복구
git clone nabis_contest_20240115.bundle nabis_contest_restored
```

## 보안 고려사항

### 1. HTTPS 설정
- GitHub Pages는 자동으로 HTTPS 제공
- 커스텀 도메인 사용 시 SSL 인증서 설정 필요

### 2. 콘텐츠 보안 정책
```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' 'unsafe-inline' https://cdn.plot.ly https://unpkg.com; 
               style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; 
               img-src 'self' data: https:; 
               connect-src 'self' https:;">
```

### 3. 데이터 보호
- 민감한 정보 포함 데이터 제외
- 공개 데이터만 사용
- 정기적인 보안 검토

## 성능 최적화

### 1. 이미지 최적화
```bash
# 이미지 압축
find . -name "*.png" -exec pngquant --ext .png --force {} \;
find . -name "*.jpg" -exec jpegoptim --max=85 {} \;
```

### 2. CSS/JS 최적화
```bash
# CSS 압축
find . -name "*.css" -exec cleancss -o {} {} \;

# JavaScript 압축
find . -name "*.js" -exec uglifyjs {} -o {} \;
```

### 3. 캐싱 전략
```html
<!-- 정적 리소스 캐싱 -->
<link rel="stylesheet" href="styles.css?v=1.0.0">
<script src="script.js?v=1.0.0"></script>
```

## 운영 체크리스트

### 일일 점검
- [ ] GitHub Pages 접근성 확인
- [ ] 주요 기능 정상 작동 확인
- [ ] 에러 로그 확인

### 주간 점검
- [ ] 성능 지표 확인
- [ ] 사용자 피드백 검토
- [ ] 보안 업데이트 확인

### 월간 점검
- [ ] 데이터 업데이트
- [ ] 백업 상태 확인
- [ ] 전체 시스템 점검
