#!/bin/bash
# GitHub Pages 테스트 실행 스크립트

echo "🚀 GitHub Pages 테스트 실행"
echo "=============================="

# 현재 디렉토리 확인
if [ ! -f "scripts/test_github_pages.py" ]; then
    echo "❌ 테스트 스크립트를 찾을 수 없습니다."
    exit 1
fi

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

# 필요한 패키지 설치 확인
echo "📦 필요한 패키지 확인 중..."
python3 -c "import requests, bs4" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  필요한 패키지를 설치합니다..."
    pip install requests beautifulsoup4
    if [ $? -ne 0 ]; then
        echo "❌ 패키지 설치 실패"
        exit 1
    fi
fi

# 테스트 실행
echo "🔍 GitHub Pages 테스트 시작..."
python3 scripts/test_github_pages.py

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 모든 테스트 통과!"
    echo "🚀 Push를 진행할 수 있습니다."
    echo ""
    echo "📁 생성된 파일:"
    echo "   • github_pages_test_results.json"
    echo "   • github_pages_test_report.md"
else
    echo ""
    echo "❌ 테스트 실패!"
    echo "🔧 문제를 수정한 후 다시 실행하세요."
    echo ""
    echo "📁 테스트 결과 파일을 확인하세요:"
    echo "   • github_pages_test_results.json"
    echo "   • github_pages_test_report.md"
    exit 1
fi
