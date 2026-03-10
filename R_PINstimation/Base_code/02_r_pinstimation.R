# =============================================================================
# [R 계산 - 샘플 검증용] PINstimation::adjpin()으로 AdjPIN 추정
# =============================================================================
#
# ■ 역할
#   Python 커스텀 MLE(02_apin_daily_00기본.py)의 APIN 결과를 PINstimation으로
#   재현·검증한다. 소수 종목 샘플 비교가 목적.
#
# ■ 병렬화 전략: "종목당 1코어" 방식
#
#   레이스 컨디션이 없는 이유:
#     - 각 워커는 독립된 Rscript 프로세스 (PSOCK 클러스터)
#     - 워커끼리 공유 메모리가 전혀 없음
#     - 각 워커는 자기 종목 데이터만 인자로 받아 결과만 반환
#     - 메인 프로세스가 결과를 수집해 합침 (쓰기 충돌 없음)
#
# ■ 메모리 추정 (32GB / 가용 23GB 기준)
#   ┌──────────────────────────────────────────────────────┐
#   │ 구성 요소                  │ 메모리                  │
#   ├──────────────────────────────────────────────────────┤
#   │ 워커 1개 (R + 패키지)     │ ~185 MB                 │
#   │ adjpin() 피크 작업 메모리  │ ~15 MB                  │
#   │ 워커 1개 합계             │ ~200 MB                 │
#   │ 16워커 합계               │ ~3,200 MB  (3.1 GB)     │
#   │ 메인 세션 + OS 예약       │ ~1,374 MB  (1.3 GB)     │
#   ├──────────────────────────────────────────────────────┤
#   │ 총 예상 피크              │ ~4,574 MB  (4.5 GB)     │
#   │ 가용 메모리               │ 23,552 MB  (23.0 GB)    │
#   │ 여유                      │ ~18,978 MB (18.5 GB)    │
#   │ 사용률                    │ ~19.4%  ✅ 매우 안전    │
#   └──────────────────────────────────────────────────────┘
#
# ■ 입력
#   R_output/sample_daily_bs.parquet  (Symbol, Date, B, S)
#
# ■ 출력
#   R_output/r_apin_results.csv      ← Excel / pandas 에서 열기 편함
#   R_output/r_apin_results.parquet  ← Python polars로 join 시 사용
#
# ■ 의존 패키지 (최초 1회 설치)
#   install.packages(c("PINstimation", "arrow", "dplyr"))
#   ※ parallel 은 R 기본 내장 패키지 — 별도 설치 불필요
#
# ■ 실행 방법
#   Rscript 02_r_pinstimation.R "E:/vpin_project_parquet/R_output"
#   또는 RStudio에서 OUTPUT_DIR 수정 후 전체 실행
#
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)   # adjpin() 함수
  library(arrow)          # parquet I/O
  library(dplyr)          # 데이터 조작
  library(parallel)       # R 기본 내장 병렬 처리 (별도 설치 불필요)
})


# =============================================================================
# ★ 사용자 설정 블록 — 여기만 수정하면 됩니다
# =============================================================================

args       <- commandArgs(trailingOnly = TRUE)
OUTPUT_DIR <- if (length(args) >= 1) args[1] else "E:/vpin_project_parquet/R_output"

# ── 롤링 윈도우 설정 ──────────────────────────────────────────────────────
WINDOW_SIZE    <- 60   # 영업일 기준 롤링 윈도우 크기
MIN_VALID_DAYS <- 30   # 윈도우 내 실제 거래일(B+S>0) 최소 개수

# ── PINstimation adjpin() 설정 ────────────────────────────────────────────
ADJPIN_METHOD      <- "ECM"   # ECM: ML보다 빠르고 수렴 안정성 높음
ADJPIN_INITIALSETS <- "GE"    # GE: Ghachem & Ersan 표준 초기값
NUM_INITIAL_SETS   <- 20      # 초기값 세트 수

# ── 병렬 처리 설정 ────────────────────────────────────────────────────────
# 기본값: 전체 논리 코어 수. 종목 수보다 크면 종목 수로 자동 제한됨.
# 수동 지정: NUM_WORKERS <- 8
NUM_WORKERS <- parallel::detectCores(logical = TRUE)


# =============================================================================
# 시작 메시지
# =============================================================================

cat(sprintf("\n%s\n", strrep("=", 65)))
cat("[R PINstimation - 샘플 검증용] 시작\n")
cat(sprintf("%s\n", strrep("=", 65)))
cat(sprintf("  OUTPUT_DIR     : %s\n", OUTPUT_DIR))
cat(sprintf("  윈도우 크기    : %d 영업일\n", WINDOW_SIZE))
cat(sprintf("  최소 유효일    : %d일\n",       MIN_VALID_DAYS))
cat(sprintf("  Method         : %s\n",         ADJPIN_METHOD))
cat(sprintf("  Initial Sets   : %s (%d sets)\n", ADJPIN_INITIALSETS, NUM_INITIAL_SETS))
cat(sprintf("  논리 코어 수   : %d\n", parallel::detectCores(logical = TRUE)))


# =============================================================================
# 1. 데이터 로드
# =============================================================================

input_path <- file.path(OUTPUT_DIR, "sample_daily_bs.parquet")

if (!file.exists(input_path)) {
  stop(sprintf(
    "[Error] 파일 없음: %s\n먼저 01_python_preprocessing.py 를 실행하세요.",
    input_path
  ))
}

cat(sprintf("\n[1] 데이터 로드: %s\n", input_path))
sample_bs       <- arrow::read_parquet(input_path)
sample_bs$Date  <- as.Date(sample_bs$Date)

symbols   <- sort(unique(sample_bs$Symbol))
n_symbols <- length(symbols)
n_workers <- min(NUM_WORKERS, n_symbols)  # 종목 수 초과로 워커 낭비 방지

cat(sprintf("  종목 수    : %d개  (%s)\n", n_symbols, paste(symbols, collapse = ", ")))
cat(sprintf("  전체 행    : %s\n",          format(nrow(sample_bs), big.mark = ",")))
cat(sprintf("  날짜 범위  : %s ~ %s\n",
            format(min(sample_bs$Date)), format(max(sample_bs$Date))))
cat(sprintf("  사용 워커  : %d개 / %d코어 (종목 1개당 워커 1개)\n",
            n_workers, parallel::detectCores(logical = TRUE)))


# =============================================================================
# 2. 워커 함수 정의
#
#    worker_process_symbol(args):
#      - 각 워커(독립 Rscript 프로세스)에서 실행되는 함수
#      - 외부 전역 변수를 참조하지 않고 args 인자만으로 완결 동작
#        → 레이스 컨디션 원천 차단
#      - 내부에 estimate_window()를 중첩 정의하여 클로저로 인자 참조
#
#    makeCluster / parLapply 동작 원리:
#      makeCluster(n, type="PSOCK"):
#        독립 Rscript 프로세스 n개를 소켓으로 연결.
#        Windows/Linux/macOS 모두 동작. 각 프로세스는 완전히 독립 메모리.
#      parLapply(cl, list, fun):
#        list의 각 원소를 워커에 1:1 분배 후 fun 실행.
#        모든 워커가 완료될 때까지 대기, 결과 리스트 반환.
#      stopCluster(cl):
#        작업 완료(또는 오류) 후 반드시 호출해 워커 프로세스 정리.
# =============================================================================

#' 단일 종목의 전체 롤링 윈도우를 처리하는 워커 함수.
#'
#' @param args  named list:
#'   $sym_df          data.frame(Date, B, S) — 해당 종목만 포함, Date 순 정렬
#'   $symbol          character              — 종목 코드 (결과 컬럼에 기록)
#'   $window_size     integer                — 롤링 윈도우 크기 (60)
#'   $min_valid_days  integer                — 유효 거래일 최소값 (30)
#'   $method          character              — adjpin method ("ECM")
#'   $initialsets     character              — adjpin initialsets ("GE")
#'   $num_init        integer                — 초기값 세트 수 (20)
#' @return  data.frame (Symbol, Date, valid_days, a~ps, APIN, PSOS)
#'          추정 성공 행만 포함; 성공 행이 없으면 0행 data.frame 반환

worker_process_symbol <- function(args) {

  # ── 인자 unpack ────────────────────────────────────────────────────────
  sym_df         <- args$sym_df
  symbol         <- args$symbol
  window_size    <- args$window_size
  min_valid_days <- args$min_valid_days
  method         <- args$method
  initialsets    <- args$initialsets
  num_init       <- args$num_init

  # 워커 프로세스는 독립 R 세션이므로 패키지를 직접 로드해야 함
  suppressPackageStartupMessages(library(PINstimation))

  n_days      <- nrow(sym_df)
  sym_results <- list()

  # 데이터 부족 시 즉시 빈 결과 반환
  if (n_days < window_size) return(data.frame())

  # ── 단일 윈도우 추정 내부 함수 (클로저로 method/initialsets/num_init 참조) ──
  #
  # PINstimation 파라미터명 ↔ Python 커스텀 코드 대응:
  #   alpha   ↔ a    (정보 이벤트 확률)      delta   ↔ d    (호재 조건부 확률)
  #   theta   ↔ t1   (무정보일 충격 확률)    theta'  ↔ t2   (정보일 충격 확률)
  #   mu_b    ↔ ub   (호재 정보 매수 도착률) mu_s    ↔ us   (악재 정보 매도 도착률)
  #   eps_b   ↔ eb   (비정보 매수 도착률)    eps_s   ↔ es   (비정보 매도 도착률)
  #   d_b     ↔ pb   (충격 추가 매수 Δ_b)   d_s     ↔ ps   (충격 추가 매도 Δ_s)
  estimate_window <- function(window_df) {
    tryCatch({
      result <- adjpin(
        data        = window_df,
        method      = method,
        initialsets = initialsets,
        num_init    = num_init,
        verbose     = FALSE
      )
      params <- result@parameters
      list(
        a  = as.numeric(params["alpha"]),  d  = as.numeric(params["delta"]),
        t1 = as.numeric(params["theta"]),  t2 = as.numeric(params["thetap"]),
        ub = as.numeric(params["mu.b"]),   us = as.numeric(params["mu.s"]),
        eb = as.numeric(params["eps.b"]),  es = as.numeric(params["eps.s"]),
        pb = as.numeric(params["d.b"]),    ps = as.numeric(params["d.s"]),
        APIN = as.numeric(result@adjpin),
        PSOS = as.numeric(result@psos),
        converged = TRUE
      )
    }, error = function(e) list(converged = FALSE))
  }

  # ── 60일 슬라이딩 윈도우 루프 ────────────────────────────────────────
  # i: 윈도우 마지막 날의 행 인덱스 (1-based R 인덱스)
  #   i = window_size → 첫 번째 완전한 윈도우 (행 1 ~ 60)
  #   i = n_days      → 마지막 윈도우 (행 n_days-59 ~ n_days)
  for (i in window_size:n_days) {

    start_idx  <- i - window_size + 1
    window_B   <- sym_df$B[start_idx:i]
    window_S   <- sym_df$S[start_idx:i]
    end_date   <- sym_df$Date[i]

    # 유효 거래일 검사 (거래 희소 윈도우 스킵)
    valid_days <- sum((window_B + window_S) > 0)
    if (valid_days < min_valid_days) next

    est <- estimate_window(data.frame(B = window_B, S = window_S))
    if (!isTRUE(est$converged)) next

    sym_results[[length(sym_results) + 1]] <- data.frame(
      Symbol     = symbol,
      Date       = as.Date(end_date),
      valid_days = valid_days,
      a  = est$a,  d  = est$d,  t1 = est$t1, t2 = est$t2,
      ub = est$ub, us = est$us, eb = est$eb,  es = est$es,
      pb = est$pb, ps = est$ps,
      APIN = est$APIN,
      PSOS = est$PSOS,
      stringsAsFactors = FALSE
    )
  }

  if (length(sym_results) == 0) return(data.frame())
  do.call(rbind, sym_results)
}


# =============================================================================
# 3. 병렬 실행
# =============================================================================

cat(sprintf("\n[2] 병렬 추정 시작 (워커=%d, 윈도우=%d일, 최소유효일=%d일)\n",
            n_workers, WINDOW_SIZE, MIN_VALID_DAYS))
cat(sprintf("%s\n", strrep("-", 65)))

# 종목이 0개면 이후 makeCluster(0) 호출로 오류 발생 → 사전 차단
if (n_symbols == 0) {
  stop("[Error] 처리할 종목이 없습니다. sample_daily_bs.parquet의 Symbol 값을 확인하세요.")
}

# 각 워커에 넘길 args 리스트 구성
# 종목별 데이터를 미리 잘라서 전달 → 워커가 전체 df를 받을 필요 없음
symbol_args <- lapply(symbols, function(sym) {
  sym_rows <- sample_bs[sample_bs$Symbol == sym, c("Date", "B", "S")]
  sym_rows <- sym_rows[order(sym_rows$Date), ]   # Date 순 정렬 보장
  list(
    sym_df         = sym_rows,
    symbol         = sym,
    window_size    = WINDOW_SIZE,
    min_valid_days = MIN_VALID_DAYS,
    method         = ADJPIN_METHOD,
    initialsets    = ADJPIN_INITIALSETS,
    num_init       = NUM_INITIAL_SETS
  )
})

start_time <- Sys.time()

# 클러스터 생성 (독립 Rscript 프로세스 n_workers개)
cl <- makeCluster(n_workers, type = "PSOCK")
# on.exit: 정상 완료·에러·중단 어떤 경우에도 워커 프로세스를 반드시 정리
on.exit(tryCatch(stopCluster(cl), error = function(e) NULL), add = TRUE)

cat(sprintf("  워커 %d개 생성 완료, 추정 실행 중...\n", n_workers))

results_list <- tryCatch({
  parLapply(cl, symbol_args, worker_process_symbol)
}, error = function(e) {
  # 에러 메시지를 출력한 뒤 스크립트를 중단 (quit 대신 stop 사용 — RStudio 세션 유지)
  stop(sprintf("[Error] 병렬 실행 중 오류: %s\n  → PINstimation 패키지 설치 여부와 데이터 형식을 확인하세요.", e$message))
})

elapsed <- difftime(Sys.time(), start_time, units = "mins")
cat(sprintf("  추정 소요 시간: %.1f분\n", as.numeric(elapsed)))


# =============================================================================
# 4. 결과 수집 및 저장
# =============================================================================

cat(sprintf("\n[3] 결과 수집 및 저장 중...\n"))

# 종목별 결과 현황 출력
for (i in seq_along(symbols)) {
  sym <- symbols[i]
  df  <- results_list[[i]]
  if (nrow(df) > 0) {
    cat(sprintf("  [%s] %d 윈도우  (APIN: %.4f ~ %.4f)\n",
                sym, nrow(df),
                min(df$APIN, na.rm = TRUE),
                max(df$APIN, na.rm = TRUE)))
  } else {
    cat(sprintf("  [%s] 추정 결과 없음\n", sym))
  }
}

# 빈 결과 제거 후 전체 병합
non_empty <- Filter(function(df) nrow(df) > 0, results_list)

if (length(non_empty) == 0) {
  stop("[Warning] 추정 결과가 없습니다. 설정값 및 데이터를 확인하세요.")
}

final_df           <- do.call(rbind, non_empty)
rownames(final_df) <- NULL
final_df           <- final_df[order(final_df$Symbol, final_df$Date), ]

# CSV 저장 (Python/Excel에서 열기 편함)
csv_path <- file.path(OUTPUT_DIR, "r_apin_results.csv")
write.csv(final_df, csv_path, row.names = FALSE)
cat(sprintf("\n  CSV 저장    : %s\n", csv_path))

# parquet 저장 (Python polars에서 직접 join 가능)
parquet_path <- file.path(OUTPUT_DIR, "r_apin_results.parquet")
arrow::write_parquet(final_df, parquet_path, compression = "zstd")
cat(sprintf("  parquet 저장: %s\n", parquet_path))


# =============================================================================
# 5. 결과 요약 출력
# =============================================================================

cat(sprintf("\n%s\n", strrep("=", 65)))
cat("[결과 요약]\n")
cat(sprintf("  전체 레코드  : %s\n", format(nrow(final_df), big.mark = ",")))
cat(sprintf("  종목 수      : %d\n", length(unique(final_df$Symbol))))
cat(sprintf("  APIN 평균    : %.4f\n", mean(final_df$APIN, na.rm = TRUE)))
cat(sprintf("  APIN 범위    : %.4f ~ %.4f\n",
            min(final_df$APIN, na.rm = TRUE),
            max(final_df$APIN, na.rm = TRUE)))

# 종목별 요약
cat("\n[종목별 요약]\n")
sym_summary <- final_df %>%
  group_by(Symbol) %>%
  summarise(
    windows    = n(),
    apin_mean  = round(mean(APIN, na.rm = TRUE), 4),
    apin_min   = round(min(APIN,  na.rm = TRUE), 4),
    apin_max   = round(max(APIN,  na.rm = TRUE), 4),
    date_start = as.character(min(Date)),
    date_end   = as.character(max(Date)),
    .groups    = "drop"
  )
print(sym_summary)

# 미리보기 (상위 10행)
cat("\n[미리보기 — 상위 10행]\n")
print(head(final_df[, c("Symbol", "Date", "a", "d", "t1", "t2",
                         "ub", "us", "eb", "es", "pb", "ps",
                         "APIN", "PSOS")], 10))

cat(sprintf("\n%s\n", strrep("=", 65)))
cat("[완료]\n")
cat(sprintf("  출력 파일:\n"))
cat(sprintf("    CSV     : %s\n", csv_path))
cat(sprintf("    parquet : %s\n", parquet_path))
cat(sprintf("\n  ※ Python 결과(apin_daily_rolling_*.parquet)와\n"))
cat(sprintf("    Symbol + Date 키로 join하여 APIN 컬럼을 비교하세요.\n"))
cat(sprintf("%s\n\n", strrep("=", 65)))
