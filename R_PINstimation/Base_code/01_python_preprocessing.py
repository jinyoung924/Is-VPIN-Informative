"""
=============================================================================
전체 파이프라인 구조 개요
틱 데이터 (월별 parquet) → 일별 B/S 집계 → R PINstimation APIN 추정
=============================================================================

이 스크립트(01)는 파이프라인의 Python 전처리 담당이며,
후속 R 스크립트(02_r_pinstimation.R)가 APIN을 최종 계산한다.
전체 파이프라인은 Python MLE(02_apin_daily_00기본.py)의 결과를
소수 샘플 종목으로 크로스체크하는 것이 목적이다.

─────────────────────────────────────────────────────────────────────────────
■ 전체 흐름 (데이터 → 파일 → 계산)
─────────────────────────────────────────────────────────────────────────────

  [원본 입력: 틱 데이터]
  E:\vpin_project_parquet\
  ├── KOR_2019\
  │   ├── 201901_ticks.parquet   ← 컬럼: Symbol, Date, LR(1=매수/-1=매도), ...
  │   ├── 201902_ticks.parquet
  │   └── ...  (월 1파일, 연간 ~12개)
  ├── KOR_2020\
  │   └── ...
  └── KOR_2021\
      └── ...

         ↓  [Step 1] preprocess_trade_data_polars()
            polars lazy scan → LR 필터 → 일별 group_by → B/S 합산
            전 종목·전 기간 처리 (수백만 행 → 수십만 행으로 압축)
            FORCE_REPROCESS_STEP1=False 이면 기존 파일 재사용(캐시)

  [중간 산출물 ①: 전 종목 캐시]
  R_output/all_daily_bs.parquet
    스키마 : Symbol(Utf8), Date(Date), B(UInt32), S(UInt32)
    내용   : 전 종목 × 전 기간 일별 매수·매도 건수
    용도   : Step 2의 입력 원본. 샘플 종목·기간을 바꿔도 재사용.
    행 수  : 종목 수 × 영업일 수 (예: 2,000종목 × 750일 ≈ 150만 행)

         ↓  [Step 2] run_step2()
            ① all_daily_bs.parquet 로드
            ② 종목 필터  : SAMPLE_SYMBOLS 16개만 추출
            ③ 기간 필터  : SAMPLE_YEARS [2019, 2020, 2021]
            ④ 영업일 캘린더 구축
               → 샘플 데이터 기준 거래가 한 번이라도 있었던 날의 합집합
               → 약 750 영업일 (3년 × 약 250일/년)
            ⑤ 종목별 캘린더 left join
               → 거래 없는 영업일은 B=S=0 으로 채움 (행 누락 방지)
               → 이 정렬이 없으면 60행 ≠ 60 영업일 문제 발생

  [중간 산출물 ②: R 입력 파일]  ← 이 스크립트의 최종 출력
  R_output/sample_daily_bs.parquet
    스키마 : Symbol(Utf8), Date(Date), B(Int32), S(Int32)
    내용   : 16개 종목 × 750 영업일 = 12,000 행 (거래 없는 날 B=S=0 포함)
    용도   : R 스크립트가 이 단일 파일만 읽어 APIN 계산 수행
    특징   : 영업일 캘린더에 완전 정렬된 상태

         ↓  [02_r_pinstimation.R 실행]
            Rscript 02_r_pinstimation.R "E:/vpin_project_parquet/R_output"

─────────────────────────────────────────────────────────────────────────────
■ R 스크립트(02) 내부 처리 흐름 상세
─────────────────────────────────────────────────────────────────────────────

  [R 입력]
  sample_daily_bs.parquet (16종목 × 750영업일 = 12,000행)

  ① 데이터 로드 및 종목 목록 추출
       symbols = 16개 종목 (알파벳순 정렬)
       n_workers = min(detectCores(), 16)  ← 코어 수와 종목 수 중 작은 값

  ② 워커 인자 사전 구성 (parLapply 전달용)
       symbol_args = [
         { sym_df: 각 종목의 750행, symbol: "AXXXXXX", window_size: 60,
           min_valid_days: 30, method: "ECM", initialsets: "GE", num_init: 20 },
         ...  (16종목 각각)
       ]
       ※ 종목 데이터를 미리 분리해서 넘김 → 워커가 전체 12,000행 불필요

  ③ 병렬 클러스터 생성 및 실행
       cl <- makeCluster(16, type="PSOCK")
         └─ 독립 Rscript 프로세스 최대 16개 생성 (Windows/Linux/macOS 호환)
            각 프로세스: 완전 독립 메모리, 공유 없음

       parLapply(cl, symbol_args, worker_process_symbol)
         └─ 16종목을 16워커에 1:1로 분배하여 동시 실행

  ④ 워커 내부: worker_process_symbol() 처리 내용 (종목당)
       library(PINstimation)  ← 워커 프로세스는 새 R 세션이므로 직접 로드

       for i in 60 → 750:          ← 60일 슬라이딩 윈도우 루프
           window = 행[i-59 : i]   ← 60행 슬라이스 (B벡터, S벡터)
           valid_days = sum(B+S > 0)
           if valid_days < 30: skip   ← 거래 희소 윈도우 제외

           adjpin(
               data        = data.frame(B=60행, S=60행),
               method      = "ECM",    ← Expectation-Conditional Maximization
               initialsets = "GE",     ← HAC 클러스터링 기반 초기값 20개
               num_init    = 20,
               verbose     = FALSE
           )
           → S4 객체에서 10개 파라미터 + APIN + PSOS 추출

       윈도우 결과 행: Symbol, Date(윈도우 마지막 날),
                        valid_days, a, d, t1, t2, ub, us, eb, es, pb, ps,
                        APIN, PSOS

       1종목당 추정 윈도우 수:
         총 750 영업일, 첫 완성 윈도우 = 60번째 날
         최대 = 750 - 60 + 1 = 691개 윈도우
         실제 = valid_days < 30 필터 후 ≈ 660~691개 (거래일 밀도 따라 다름)
         단, 저유동성 종목(A001000, A047770 등)은 유효 윈도우가 크게 줄어들 수 있음

  ⑤ 메인 프로세스가 16워커 결과 수집
       results_list = [16개 종목 결과 df]
       stopCluster(cl)
       final_df = rbind(16개 df)
         ← 전체 약 8,000~11,000행 (저유동성 종목 스킵 고려)
       Symbol + Date 기준 정렬

─────────────────────────────────────────────────────────────────────────────
■ 최종 출력 파일 (R 스크립트 생성)
─────────────────────────────────────────────────────────────────────────────

  R_output/r_apin_results.csv      ← Excel / pandas 에서 열기 편함
  R_output/r_apin_results.parquet  ← Python polars로 join 시 사용

  공통 스키마 (컬럼 15개):
    Symbol     : 종목코드 (예: "A005930")
    Date       : 윈도우 마지막 날짜 (= 해당 APIN의 추정 기준일)
    valid_days : 윈도우 내 실제 거래일 수 (품질 지표, 30~60)
    a          : alpha  — 정보 이벤트 발생 확률      (Python: a)
    d          : delta  — 호재 조건부 확률            (Python: d)
    t1         : theta  — 무정보일 SPOS 충격 확률     (Python: t1)
    t2         : theta' — 정보일 SPOS 충격 확률       (Python: t2)
    ub         : mu_b   — 호재 정보 매수 추가 도착률  (Python: ub)
    us         : mu_s   — 악재 정보 매도 추가 도착률  (Python: us)
    eb         : eps_b  — 비정보 매수 기본 도착률     (Python: eb)
    es         : eps_s  — 비정보 매도 기본 도착률     (Python: es)
    pb         : d_b    — SPOS 충격 추가 매수량 Δ_b  (Python: pb)
    ps         : d_s    — SPOS 충격 추가 매도량 Δ_s  (Python: ps)
    APIN       : Adjusted PIN (핵심 출력값)
    PSOS       : Probability of Symmetric Order-flow Shock

─────────────────────────────────────────────────────────────────────────────
■ APIN 검증용 16개 종목 포트폴리오 시나리오
─────────────────────────────────────────────────────────────────────────────

  각 종목은 서로 다른 APIN 패턴을 유도하도록 설계되었다.
  결과 해석 시 아래 예상값과 비교하여 모델이 올바로 동작하는지 확인한다.

  ┌─────┬────────────┬──────────────────────┬──────────────────────────────────────┐
  │그룹 │ 코드       │ 종목명               │ 예상 APIN 패턴                       │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 1   │ A005930    │ 삼성전자             │ APIN 최저·안정. α 작음.              │
  │ 대형│ A105560    │ KB금융               │ APIN 낮음. α 매우 작음.              │
  │우량주│           │                      │ 거시경제 반응, 내부자 진입 희소.     │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 2   │ A068270    │ 셀트리온             │ APIN 중~높음. 임상·공매도 시기        │
  │바이오│            │                      │ α, μ 동시 급등 관찰 기대.            │
  │     │ A028300    │ HLB(에이치엘비)      │ APIN 높음. FDA 이슈 윈도우에서       │
  │     │            │                      │ α·μ 극단값, 수렴 실패 가능성 있음.  │
  │     │ A019170    │ 신풍제약             │ 2020~2021 코로나 테마 폭등·폭락.     │
  │     │            │                      │ 극단적 파라미터 변화 관찰용.         │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 3   │ A086520    │ 에코프로             │ APIN보다 PSOS 높을 것으로 예상.      │
  │밈/  │            │                      │ Δ(pb/ps) 크고 α 상대적으로 낮음.    │
  │개인 │ A005490    │ POSCO홀딩스          │ 2차전지 내러티브 전환 구간에서        │
  │수급 │            │                      │ 유동성 충격(Δ) 파라미터 급등 예상.  │
  │     │ A022100    │ 포스코DX             │ 코스피 이전 상장 전후 패시브 자금     │
  │     │            │                      │ 유입 → 비정보성 Δ 급등, α 낮음.    │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 4   │ A352820    │ 하이브               │ 2020년 10월 IPO. 상장 초기           │
  │엔터/│            │                      │ 정보 비대칭 → α 높음 예상.           │
  │게임 │ A036570    │ 엔씨소프트           │ 신작 실적 쇼크 시 μ_s 급등,         │
  │     │            │                      │ 대규모 매도 비대칭 관찰용.           │
  │     │ A035720    │ 카카오               │ 규제·자회사 이벤트 다양. α 중간,    │
  │     │            │                      │ 이벤트 전후 시계열 변화 추적용.      │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 5   │ A011200    │ HMM                  │ 해운 운임 폭등 구간 APIN 변화.       │
  │턴어 │            │                      │ 흑자 전환 시점 전후 α 비교용.        │
  │라운드│ A042660   │ 한화오션(구 대우조선)│ 매각·파업 노이즈 → α vs Δ 분리      │
  │     │            │                      │ 가능 여부 테스트용.                  │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 6   │ A001000    │ 신라섬유             │ 저유동성 → 유효 윈도우 극소.         │
  │초저 │            │                      │ 수렴 실패·NULL 비율 측정용.          │
  │유동성│ A047770   │ 코데즈컴바인         │ 품절주 숏스퀴즈 이력. 거래 패턴      │
  │     │            │                      │ 기형적 → 알고리즘 예외 처리 확인용. │
  ├─────┼────────────┼──────────────────────┼──────────────────────────────────────┤
  │ 7   │ A001440    │ 대한전선             │ 액면병합·유상증자·무상감자 빈번.     │
  │자본 │            │                      │ 자본 이벤트 전후 B/S 급변 → 모델    │
  │이벤트│           │                      │ 노이즈 내성 테스트용.                │
  └─────┴────────────┴──────────────────────┴──────────────────────────────────────┘

  ※ APIN 파라미터 해석 요약:
       α (alpha)   : 정보 이벤트 발생 확률. 높을수록 내부자 거래 빈번.
       μ (mu_b/s)  : 정보 이벤트 발생 시 추가 매수/매도 강도.
       Δ (pb/ps)   : 비정보성 유동성 충격 크기. 밈주·패시브에서 높음.
       PSOS        : 대칭 주문 충격 확률. Δ 높은 종목에서 APIN과 괴리.

  ※ 주의사항:
       - A352820(하이브)은 2020년 10월 상장 → 2019년 데이터 없음
       - A086520(에코프로) 2차전지 테마 급등은 2023년 이후가 주요 구간이나
         2019~2021 기간에도 초기 상승 패턴 관찰 가능
       - A001000, A047770은 유효 윈도우가 극히 적을 수 있으며
         MIN_VALID_DAYS=30 조건으로 대부분 스킵될 가능성 있음
         → 수렴 실패 NULL 비율 자체가 측정 지표

─────────────────────────────────────────────────────────────────────────────
■ Python 결과와 비교하는 방법 (검증 워크플로)
─────────────────────────────────────────────────────────────────────────────

  import polars as pl

  py = pl.read_parquet("apin_daily_rolling_*.parquet")   # Python MLE 결과
  r  = pl.read_parquet("R_output/r_apin_results.parquet")

  compare = py.join(r, on=["Symbol", "Date"], how="inner", suffix="_r")
  # APIN vs APIN_r 컬럼 비교
  # 두 방식의 차이는 초기값 전략 차이(Python: 그리드서치 vs R: GE HAC)에서 기인

─────────────────────────────────────────────────────────────────────────────
■ 이 스크립트(01)의 역할 요약
─────────────────────────────────────────────────────────────────────────────

  기존 구조 대비 변경점:
    - [제거] Step 3 롤링 윈도우 생성 → 60배 데이터 증폭 문제 해소
    - [추가] SAMPLE_SYMBOLS : 검증 대상 종목 리스트 직접 지정 (16개)
    - [추가] SAMPLE_YEARS   : 검증 기간(연도) 직접 지정
    - [변경] 출력물이 sample_daily_bs.parquet 단 하나로 단순화

  파일 흐름 요약:
    [월별 틱 parquet N개]
        → (Step 1) all_daily_bs.parquet        ← 전 종목 캐시
        → (Step 2) sample_daily_bs.parquet     ← R 입력 (16종목)
        → (R 병렬) r_apin_results.parquet      ← 최종 APIN 결과

  실행 순서:
    1) python 01_python_preprocessing.py
    2) Rscript 02_r_pinstimation.R "E:/vpin_project_parquet/R_output"
=============================================================================
"""

import os
import sys
import glob
import numpy as np
import polars as pl
import warnings
from datetime import datetime
from typing import List, Optional
from tqdm import tqdm

# Windows cp949 환경에서 Polars DataFrame 출력 시 유니코드 오류 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# ★ 사용자 설정 블록 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR   = r"E:\vpin_project_parquet"      # 월별 틱 parquet 루트 경로
OUTPUT_DIR = os.path.join(BASE_DIR, "R_output")  # 출력 경로

# ── 검증 대상 종목 (16개) ─────────────────────────────────────────────────
# 그룹 설계 의도: 서로 다른 APIN 패턴(낮음/중간/높음/수렴실패)을 유도하여
# Python MLE와 R PINstimation의 동작을 다양한 시나리오에서 비교·검증한다.
# 각 종목의 예상 APIN 패턴은 상단 독스트링 포트폴리오 시나리오 표 참조.
#
SAMPLE_SYMBOLS: Optional[List[str]] = [
    # ── 그룹 1: 대형 우량주 (Baseline / 대조군) ─────────────────────────
    # 예상: APIN 최저·안정. α 작음. 모델 수렴 가장 안정적.
    "005930",   # 삼성전자      KOSPI 시총 1위, 거래량 압도적
    "105560",   # KB금융        대표 가치주·배당주. α 매우 낮을 것으로 예상.

    # ── 그룹 2: 바이오/제약 (High α, High μ) ────────────────────────────
    # 예상: 임상·FDA 이슈 구간에서 α·μ 동시 급등. 수렴 실패 가능성 있음.
    "068270",   # 셀트리온      잦은 공매도 타겟, 임상 발표 비대칭성 검증
    "028300",   # HLB           FDA 신약 승인 이슈, 극단적 α·μ 확인용
    "019170",   # 신풍제약      2020~2021 코로나 테마 폭등·폭락, 극단 파라미터 관찰

    # ── 그룹 3: 밈/개인수급 (High Δ, PSOS 높음) ────────────────────────
    # 예상: α보다 Δ(pb/ps)가 높게 잡힘. PSOS > APIN 역전 가능성.
    "086520",   # 에코프로      2차전지 광풍 대장주. 비정보성 유동성 충격 극대화.
    "005490",   # POSCO홀딩스   철강→2차전지 내러티브 전환 구간 Δ 급등 예상
    "022100",   # 포스코DX      코스피 이전상장 전후 패시브 자금 유입 → 비정보성 Δ

    # ── 그룹 4: 엔터/게임 (돌발 변수, 실적 쇼크) ────────────────────────
    # 예상: 이벤트 발생 윈도우에서 α 급등 후 평균 회귀. 이벤트 시계열 추적용.
    "352820",   # 하이브        2020년 10월 IPO. 상장 초기 정보 비대칭 케이스.
    "036570",   # 엔씨소프트    신작 실적 쇼크 시 μ_s 급등, 매도 비대칭 관찰
    "035720",   # 카카오        규제·자회사 이벤트 다양. α 이벤트 전후 비교용.

    # ── 그룹 5: 턴어라운드/거시경제 민감주 (구조적 변화) ────────────────
    # 예상: 흑자 전환·매각 이슈 구간에서 α vs Δ 분리 가능 여부 확인.
    "011200",   # HMM           해운 운임 폭등 → 흑자 전환 전후 APIN 비교
    "042660",   # 한화오션      구 대우조선해양. 파업·매각 노이즈 테스트

    # ── 그룹 6: 초저유동성/품절주 (수렴·알고리즘 한계 테스트) ──────────
    # 예상: 유효 윈도우 극소 → 수렴 실패 NULL 비율 자체가 측정 지표.
    "001000",   # 신라섬유      대표 저유동성 품절주. 하루 거래 수건 수준.
    "047770",   # 코데즈컴바인  과거 숏스퀴즈·상한가 이력. 기형적 B/S 패턴.

    # ── 그룹 7: 잦은 자본 이벤트 (노이즈 내성 테스트) ──────────────────
    # 예상: 액면병합·유상증자·무상감자 전후 B/S 급변 → 모델 노이즈 반응 확인.
    "001440",   # 대한전선      자본 이벤트 빈번. 주식수·가격 물리적 변동 종목.
]

# ── 검증 기간 (연도 리스트) ────────────────────────────────────────────────
# None 이면 전체 기간. 최근 3년 예시: [2019, 2020, 2021]
SAMPLE_YEARS: Optional[List[int]] = [2019, 2020, 2021]

# ── 기타 설정 ──────────────────────────────────────────────────────────────
# True: all_daily_bs.parquet가 있어도 틱 데이터부터 다시 집계
FORCE_REPROCESS_STEP1 = False

# Step 1에서 탐색할 연도 폴더 (None 이면 KOR_* 전체 자동 탐색)
# SAMPLE_YEARS와 별개 — 전체 기간을 한 번 집계해 캐시해 두면 효율적
YEAR_FOLDERS: Optional[List[str]] = None


# =============================================================================
# [Step 1] 틱 데이터 → 일별 B/S 집계
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일(월별 틱 데이터)을 읽어 일별 매수/매도 건수로 집계한다.

    원본 컬럼 LR: 1 → 매수(Buy), -1 → 매도(Sell)
    반환 스키마: Symbol (Utf8), Date (pl.Date), B (UInt32), S (UInt32)
    """
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    aggregated_df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "LR"])
        .filter(pl.col("LR").is_in([1, -1]))           # 매수·매도만 남김
        .with_columns([
            (pl.col("LR") == 1).cast(pl.UInt32).alias("is_Buy"),
            (pl.col("LR") == -1).cast(pl.UInt32).alias("is_Sell"),
        ])
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("is_Buy").sum().alias("B"),
            pl.col("is_Sell").sum().alias("S"),
        ])
        .filter((pl.col("B") > 0) | (pl.col("S") > 0))  # 거래 없는 날 제외
        .sort(["Symbol", "Date"])
        .collect()
    )

    # Date 타입 강제 변환 (Datetime으로 들어올 경우 대비)
    if aggregated_df["Date"].dtype != pl.Date:
        aggregated_df = aggregated_df.with_columns(pl.col("Date").cast(pl.Date))

    print(f"{aggregated_df.height:,} records")
    return aggregated_df


def get_parquet_files(base_dir: str, year_folders: Optional[List[str]]) -> List[str]:
    """base_dir 아래 KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다."""
    if year_folders is None:
        year_folders = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("KOR_")
        ])

    parquet_files = []
    for yf in year_folders:
        folder_path = os.path.join(base_dir, yf)
        if not os.path.exists(folder_path):
            print(f"[Warning] 폴더 없음: {folder_path}")
            continue
        files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        parquet_files.extend(files)

    print(f"\n[파일 탐색]")
    print(f"  기준 경로   : {base_dir}")
    print(f"  대상 연도   : {year_folders if year_folders else 'All'}")
    print(f"  parquet 파일: {len(parquet_files)}개")
    if parquet_files:
        print(f"  첫 번째     : {os.path.basename(parquet_files[0])}")
        print(f"  마지막      : {os.path.basename(parquet_files[-1])}")

    return parquet_files


def run_step1(base_dir: str, year_folders: Optional[List[str]], output_dir: str) -> str:
    """
    Step 1: 월별 틱 parquet → all_daily_bs.parquet

    FORCE_REPROCESS_STEP1=False이고 파일이 존재하면 기존 파일을 그대로 반환.
    전 종목·전 기간을 한 번만 집계해 캐시해 두면, 이후 샘플 변경 시 Step 1을 재실행하지 않아도 된다.

    반환값: 저장된 all_daily_bs.parquet 경로 (실패 시 빈 문자열)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_daily_bs.parquet")

    # 캐시 히트 — 기존 파일 재사용
    if not FORCE_REPROCESS_STEP1 and os.path.exists(output_path):
        print(f"\n{'='*65}")
        print(f"[Step 1 스킵] 기존 집계 파일 재사용: {output_path}")
        print(f"  (재생성하려면 FORCE_REPROCESS_STEP1 = True 로 설정하세요)")
        print(f"{'='*65}\n")
        return output_path

    parquet_files = get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("\n[Error] parquet 파일을 찾을 수 없습니다.")
        return ""

    print(f"\n{'='*65}")
    print("[Step 1] 틱 데이터 → 일별 B/S 집계")
    print(f"{'='*65}\n")

    all_dfs = []
    skipped = 0
    for path in parquet_files:
        df = preprocess_trade_data_polars(path)
        if df.is_empty():
            print(f"    → [Warning] 데이터 없음, 건너뜀")
            skipped += 1
        else:
            all_dfs.append(df)

    if not all_dfs:
        print("\n[Error] 유효한 데이터가 없습니다.")
        return ""

    # 여러 파일에 걸친 동일 Symbol+Date 중복 합산
    full_df = (
        pl.concat(all_dfs, how="vertical")
        .group_by(["Symbol", "Date"])
        .agg([pl.col("B").sum(), pl.col("S").sum()])
        .sort(["Symbol", "Date"])
    )

    full_df.write_parquet(output_path, compression="zstd")

    print(f"\n[Step 1 완료]")
    print(f"  저장 경로  : {output_path}")
    print(f"  전체 행 수 : {full_df.height:,}")
    print(f"  종목 수    : {full_df['Symbol'].n_unique():,}")
    print(f"  날짜 범위  : {full_df['Date'].min()} ~ {full_df['Date'].max()}")
    print(f"  처리/스킵  : {len(parquet_files) - skipped} / {skipped}개")
    print(f"{'='*65}\n")

    return output_path


# =============================================================================
# [Step 2] 샘플 필터 + 영업일 캘린더 정렬 → sample_daily_bs.parquet
# =============================================================================

def run_step2(
    daily_bs_path: str,
    output_dir: str,
    sample_symbols: Optional[List[str]],
    sample_years: Optional[List[int]],
) -> str:
    """
    Step 2: 전체 B/S에서 샘플 종목·기간 추출 → 영업일 캘린더 정렬 → sample_daily_bs.parquet

    처리 순서:
      1. all_daily_bs.parquet 로드
      2. 종목 필터 (sample_symbols)
      3. 연도 필터 (sample_years)
      4. 시장 공통 영업일 캘린더 추출
         ※ 필터링된 데이터 기준으로 캘린더를 잡으므로
           샘플 기간 외 영업일이 끼어들지 않는다
      5. 각 종목을 캘린더에 left join → 거래 없는 영업일은 B=S=0

    왜 캘린더 정렬이 필요한가:
      특정 종목이 어떤 날 거래가 없으면 틱 데이터에 해당 행 자체가 없다.
      단순히 "과거 60개 관측치"로 롤링 윈도우를 잡으면 실제로 수개월에 걸친
      창을 60 거래일로 오해하는 오류가 발생한다.
      영업일 캘린더에 정렬하면 60행이 항상 정확히 60 영업일을 의미한다.

    반환값: 저장된 sample_daily_bs.parquet 경로
    """
    print(f"\n{'='*65}")
    print("[Step 2] 샘플 필터 + 영업일 캘린더 정렬")
    print(f"{'='*65}")

    # ── 1. 전체 집계 데이터 로드 ──────────────────────────────────────────
    daily_bs = pl.read_parquet(daily_bs_path)
    print(f"  [로드] 전체 : {daily_bs['Symbol'].n_unique():,} 종목 / {daily_bs.height:,} 행")

    # ── 2. 종목 필터 ──────────────────────────────────────────────────────
    if sample_symbols:
        # 지정 종목 중 실제 데이터에 존재하는 것만 처리
        available = set(daily_bs["Symbol"].unique().to_list())
        missing   = [s for s in sample_symbols if s not in available]
        if missing:
            print(f"  [Warning] 데이터 없는 종목 무시: {missing}")
        valid_symbols = [s for s in sample_symbols if s in available]
        if not valid_symbols:
            raise ValueError("[Error] 유효한 종목이 없습니다. SAMPLE_SYMBOLS 확인 필요.")
        daily_bs = daily_bs.filter(pl.col("Symbol").is_in(valid_symbols))
        print(f"  [종목 필터] {valid_symbols}  → {daily_bs.height:,} 행 남음")

    # ── 3. 연도 필터 ──────────────────────────────────────────────────────
    if sample_years:
        daily_bs = daily_bs.filter(pl.col("Date").dt.year().is_in(sample_years))
        print(f"  [기간 필터] {sample_years}년  → {daily_bs.height:,} 행 남음")

    if daily_bs.is_empty():
        raise ValueError("[Error] 필터 후 데이터가 없습니다. 설정값 확인 필요.")

    print(f"\n  처리 대상 종목 : {daily_bs['Symbol'].n_unique()}개")
    print(f"  처리 대상 기간 : {daily_bs['Date'].min()} ~ {daily_bs['Date'].max()}")

    # ── 4. 시장 공통 영업일 캘린더 추출 ───────────────────────────────────
    # 샘플 데이터 기준 — 한 종목이라도 거래가 있었던 날의 합집합
    market_calendar = (
        daily_bs.select("Date").unique().sort("Date").get_column("Date")
    )
    calendar_df = pl.DataFrame({"Date": market_calendar})
    print(f"  영업일 수      : {len(market_calendar)}일")

    # ── 5. 각 종목을 영업일 캘린더에 정렬 (거래 없는 날 B=S=0) ──────────
    print("\n  캘린더 정렬 중...")
    symbols = daily_bs["Symbol"].unique().sort().to_list()

    aligned_parts = []
    for sym in tqdm(symbols, desc="  Aligning"):
        sym_df = daily_bs.filter(pl.col("Symbol") == sym).select(["Date", "B", "S"])
        aligned = (
            calendar_df
            .join(sym_df, on="Date", how="left")
            .with_columns([
                pl.lit(sym).alias("Symbol"),
                pl.col("B").fill_null(0).cast(pl.Int32),
                pl.col("S").fill_null(0).cast(pl.Int32),
            ])
            .select(["Symbol", "Date", "B", "S"])
        )
        aligned_parts.append(aligned)

    sample_df = (
        pl.concat(aligned_parts, how="vertical")
        .sort(["Symbol", "Date"])
    )

    # ── 저장 ──────────────────────────────────────────────────────────────
    output_path = os.path.join(output_dir, "sample_daily_bs.parquet")
    sample_df.write_parquet(output_path, compression="zstd")

    print(f"\n[Step 2 완료]")
    print(f"  저장 경로   : {output_path}")
    print(f"  총 행 수    : {sample_df.height:,}  "
          f"({len(symbols)} 종목 × {len(market_calendar)} 영업일)")
    print(f"\n  [종목별 요약]")
    summary = (
        sample_df
        .group_by("Symbol")
        .agg([
            pl.len().alias("calendar_days"),                              # 캘린더 전체 일수
            pl.col("B").filter(pl.col("B") + pl.col("S") > 0)
              .count().alias("actual_trade_days"),                        # 실제 거래일 수
            (pl.col("B") + pl.col("S")).mean().alias("avg_daily_volume"), # 일평균 거래량
        ])
        .sort("Symbol")
    )
    print(summary)
    print(f"{'='*65}\n")

    return output_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[Python 전처리 - 샘플용] 시작: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    n = len(SAMPLE_SYMBOLS) if SAMPLE_SYMBOLS else "전체"
    print(f"  대상 종목  : {n}개 {SAMPLE_SYMBOLS or ''}")
    print(f"  대상 기간  : {SAMPLE_YEARS or '전체'}")
    print(f"{'='*65}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 틱 데이터 → 전 종목 일별 B/S 집계 (캐시 포함)
    daily_bs_path = run_step1(
        base_dir=BASE_DIR,
        year_folders=YEAR_FOLDERS,
        output_dir=OUTPUT_DIR,
    )

    if not daily_bs_path or not os.path.exists(daily_bs_path):
        print("\n[Error] Step 1 실패. 종료합니다.")
        exit(1)

    # Step 2: 샘플 필터 + 캘린더 정렬 → sample_daily_bs.parquet
    sample_path = run_step2(
        daily_bs_path=daily_bs_path,
        output_dir=OUTPUT_DIR,
        sample_symbols=SAMPLE_SYMBOLS,
        sample_years=SAMPLE_YEARS,
    )

    elapsed = datetime.now() - start
    print(f"[전처리 완료] 소요 시간: {elapsed}")
    print(f"\n다음 단계: R에서 02_r_pinstimation.R 실행")
    print(f"  Rscript 02_r_pinstimation.R \"{OUTPUT_DIR}\"")
    print(f"  (또는 RStudio에서 OUTPUT_DIR 수정 후 실행)\n")