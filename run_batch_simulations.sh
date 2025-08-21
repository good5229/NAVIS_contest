#!/bin/bash

# BDS 시뮬레이션 배치 실행 스크립트
# 다양한 파라미터 조합으로 시뮬레이션을 실행하고 결과를 정리

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 시뮬레이션 실행 함수
run_simulation() {
    local scenario_name=$1
    local improvement_rate=$2
    local potential_multiplier=$3
    local diminishing_factor=$4
    local investment_budget=$5
    local description=$6
    local extra_args=$7
    
    log_info "시뮬레이션 실행: $scenario_name"
    log_info "파라미터: 개선률=$improvement_rate, 잠재력배수=$potential_multiplier, 체감계수=$diminishing_factor, 예산=${investment_budget}억원"
    
    # Python 스크립트 실행
    python3 simulation_runner.py \
        --output_prefix "$scenario_name" \
        --improvement_rate "$improvement_rate" \
        --potential_multiplier "$potential_multiplier" \
        --diminishing_factor "$diminishing_factor" \
        --investment_budget "$investment_budget" \
        --description "$description" \
        $extra_args
    
    if [ $? -eq 0 ]; then
        log_success "시뮬레이션 완료: $scenario_name"
        return 0
    else
        log_error "시뮬레이션 실패: $scenario_name"
        return 1
    fi
}

# 메인 실행 함수
main() {
    log_info "BDS 시뮬레이션 배치 실행 시작"
    
    # 결과 요약을 위한 배열
    declare -a successful_scenarios=()
    declare -a failed_scenarios=()
    
    # 실행 시간 측정 시작
    start_time=$(date +%s)
    
    # 1. 기본 시나리오들
    log_info "=== 기본 시나리오 실행 ==="
    
    # 보수적 시나리오
    if run_simulation "scenario_01_conservative" 0.02 0.01 0.3 800 "보수적 개선 시나리오 - 낮은 개선률과 높은 체감효과"; then
        successful_scenarios+=("scenario_01_conservative")
    else
        failed_scenarios+=("scenario_01_conservative")
    fi
    
    # 기본 시나리오
    if run_simulation "scenario_02_baseline" 0.05 0.03 0.2 1000 "기본 시나리오 - 중간 수준의 개선률"; then
        successful_scenarios+=("scenario_02_baseline")
    else
        failed_scenarios+=("scenario_02_baseline")
    fi
    
    # 적극적 시나리오
    if run_simulation "scenario_03_aggressive" 0.08 0.05 0.15 1500 "적극적 개선 시나리오 - 높은 개선률과 낮은 체감효과"; then
        successful_scenarios+=("scenario_03_aggressive")
    else
        failed_scenarios+=("scenario_03_aggressive")
    fi
    
    # 2. 개선률 변화 실험
    log_info "=== 개선률 변화 실험 ==="
    
    improvement_rates=(0.01 0.03 0.07 0.10)
    for rate in "${improvement_rates[@]}"; do
        scenario_name="experiment_rate_$(printf "%.0f" $(echo "$rate * 100" | bc))"
        description="개선률 ${rate} 실험"
        
        if run_simulation "$scenario_name" "$rate" 0.03 0.2 1000 "$description"; then
            successful_scenarios+=("$scenario_name")
        else
            failed_scenarios+=("$scenario_name")
        fi
    done
    
    # 3. 잠재력 배수 변화 실험
    log_info "=== 잠재력 배수 변화 실험 ==="
    
    multipliers=(0.01 0.02 0.04 0.06)
    for mult in "${multipliers[@]}"; do
        scenario_name="experiment_mult_$(printf "%.0f" $(echo "$mult * 100" | bc))"
        description="잠재력 배수 ${mult} 실험"
        
        if run_simulation "$scenario_name" 0.05 "$mult" 0.2 1000 "$description"; then
            successful_scenarios+=("$scenario_name")
        else
            failed_scenarios+=("$scenario_name")
        fi
    done
    
    # 4. 투자 예산 변화 실험
    log_info "=== 투자 예산 변화 실험 ==="
    
    budgets=(500 750 1250 2000)
    for budget in "${budgets[@]}"; do
        scenario_name="experiment_budget_${budget}"
        description="${budget}억원 투자 예산 실험"
        
        if run_simulation "$scenario_name" 0.05 0.03 0.2 "$budget" "$description"; then
            successful_scenarios+=("$scenario_name")
        else
            failed_scenarios+=("$scenario_name")
        fi
    done
    
    # 5. 체감 효과 변화 실험
    log_info "=== 체감 효과 변화 실험 ==="
    
    diminishing_factors=(0.1 0.25 0.35 0.5)
    for factor in "${diminishing_factors[@]}"; do
        scenario_name="experiment_diminishing_$(printf "%.0f" $(echo "$factor * 100" | bc))"
        description="체감 효과 계수 ${factor} 실험"
        
        if run_simulation "$scenario_name" 0.05 0.03 "$factor" 1000 "$description"; then
            successful_scenarios+=("$scenario_name")
        else
            failed_scenarios+=("$scenario_name")
        fi
    done
    
    # 6. 고속 실행 (시각화 건너뛰기)
    log_info "=== 고속 실행 실험 ==="
    
    fast_scenarios=(
        "fast_01:0.04:0.025:0.18:900:고속실행1"
        "fast_02:0.06:0.035:0.22:1100:고속실행2"
        "fast_03:0.09:0.045:0.16:1300:고속실행3"
    )
    
    for scenario_config in "${fast_scenarios[@]}"; do
        IFS=':' read -r name rate mult dim budget desc <<< "$scenario_config"
        
        if run_simulation "$name" "$rate" "$mult" "$dim" "$budget" "$desc" "--skip_visualization"; then
            successful_scenarios+=("$name")
        else
            failed_scenarios+=("$name")
        fi
    done
    
    # 실행 시간 측정 종료
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    
    # 결과 요약
    log_info "=== 배치 실행 완료 ==="
    log_success "실행 시간: ${execution_time}초"
    log_success "성공한 시나리오 수: ${#successful_scenarios[@]}"
    log_error "실패한 시나리오 수: ${#failed_scenarios[@]}"
    
    if [ ${#successful_scenarios[@]} -gt 0 ]; then
        log_info "성공한 시나리오:"
        for scenario in "${successful_scenarios[@]}"; do
            echo "  ✅ $scenario"
        done
    fi
    
    if [ ${#failed_scenarios[@]} -gt 0 ]; then
        log_warning "실패한 시나리오:"
        for scenario in "${failed_scenarios[@]}"; do
            echo "  ❌ $scenario"
        done
    fi
    
    # 결과 비교 보고서 생성
    generate_comparison_report "${successful_scenarios[@]}"
}

# 결과 비교 보고서 생성 함수
generate_comparison_report() {
    local scenarios=("$@")
    
    if [ ${#scenarios[@]} -eq 0 ]; then
        log_warning "성공한 시나리오가 없어 비교 보고서를 생성할 수 없습니다."
        return
    fi
    
    log_info "결과 비교 보고서 생성 중..."
    
    # 비교 보고서 생성 Python 스크립트 실행
    python3 -c "
import json
import pandas as pd
from pathlib import Path
import sys

scenarios = ['${scenarios[*]}']
scenarios = [s for s in scenarios if s.strip()]

print('=== 시뮬레이션 결과 비교 보고서 ===\\n')

results = []
for scenario in scenarios:
    try:
        params_file = Path('outputs_timeseries') / scenario / 'simulation_parameters.json'
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            result = {
                'scenario': scenario,
                'improvement_rate': params.get('improvement_rate', 0),
                'potential_multiplier': params.get('potential_multiplier', 0),
                'diminishing_factor': params.get('diminishing_factor', 0),
                'investment_budget': params.get('investment_budget', 0),
                'description': params.get('description', ''),
                'timestamp': params.get('timestamp', '')
            }
            results.append(result)
    except Exception as e:
        print(f'Warning: {scenario} 파라미터 로드 실패: {e}')

if results:
    df = pd.DataFrame(results)
    print('시나리오별 파라미터 요약:')
    print(df[['scenario', 'improvement_rate', 'potential_multiplier', 'investment_budget']].to_string(index=False))
    print()
    
    # CSV로 저장
    summary_file = Path('outputs_timeseries') / 'batch_simulation_summary.csv'
    df.to_csv(summary_file, index=False)
    print(f'비교 결과가 {summary_file}에 저장되었습니다.')
else:
    print('비교할 결과가 없습니다.')
"
    
    log_success "비교 보고서 생성 완료"
}

# 도움말 표시 함수
show_help() {
    echo "BDS 시뮬레이션 배치 실행 스크립트"
    echo
    echo "사용법:"
    echo "  $0                 # 전체 배치 실행"
    echo "  $0 --help         # 도움말 표시"
    echo "  $0 --quick        # 빠른 실행 (일부 시나리오만)"
    echo "  $0 --clean        # 이전 결과 정리"
    echo
    echo "생성되는 결과:"
    echo "  - outputs_timeseries/scenario_*    # 각 시나리오별 결과 폴더"
    echo "  - outputs_timeseries/batch_simulation_summary.csv  # 비교 요약"
    echo
}

# 이전 결과 정리 함수
clean_previous_results() {
    log_info "이전 시뮬레이션 결과 정리 중..."
    
    if [ -d "outputs_timeseries" ]; then
        # 시나리오 폴더들만 삭제 (기본 파일들은 유지)
        find outputs_timeseries -maxdepth 1 -type d -name "scenario_*" -exec rm -rf {} +
        find outputs_timeseries -maxdepth 1 -type d -name "experiment_*" -exec rm -rf {} +
        find outputs_timeseries -maxdepth 1 -type d -name "fast_*" -exec rm -rf {} +
        
        # 배치 실행 요약 파일 삭제
        rm -f outputs_timeseries/batch_simulation_summary.csv
        
        log_success "이전 결과 정리 완료"
    else
        log_info "정리할 이전 결과가 없습니다."
    fi
}

# 빠른 실행 함수
quick_run() {
    log_info "빠른 실행 모드 - 핵심 시나리오만 실행"
    
    declare -a successful_scenarios=()
    declare -a failed_scenarios=()
    
    # 핵심 시나리오들만 실행
    scenarios=(
        "quick_conservative:0.02:0.02:0.3:800:빠른실행-보수적"
        "quick_baseline:0.05:0.03:0.2:1000:빠른실행-기본"
        "quick_aggressive:0.08:0.05:0.15:1500:빠른실행-적극적"
    )
    
    for scenario_config in "${scenarios[@]}"; do
        IFS=':' read -r name rate mult dim budget desc <<< "$scenario_config"
        
        if run_simulation "$name" "$rate" "$mult" "$dim" "$budget" "$desc" "--skip_validation"; then
            successful_scenarios+=("$name")
        else
            failed_scenarios+=("$name")
        fi
    done
    
    log_success "빠른 실행 완료: 성공 ${#successful_scenarios[@]}개, 실패 ${#failed_scenarios[@]}개"
    
    if [ ${#successful_scenarios[@]} -gt 0 ]; then
        generate_comparison_report "${successful_scenarios[@]}"
    fi
}

# 스크립트 인수 처리
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --clean)
        clean_previous_results
        exit 0
        ;;
    --quick|-q)
        quick_run
        exit 0
        ;;
    "")
        main
        ;;
    *)
        log_error "알 수 없는 옵션: $1"
        show_help
        exit 1
        ;;
esac 