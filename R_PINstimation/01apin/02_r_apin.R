# =============================================================================
# [APIN 전체계산] PINstimation::adjpin() 롤링 추정
# =============================================================================
#
# 모델: Duarte & Young(2009) — Adjusted PIN (APIN)
#   표준 PIN의 비정보성 주문흐름 충격(SPOS)을 분리해 10개 파라미터로 확장.
#   핵심 지표: APIN (Adjusted PIN), PSOS (Prob. of Symmetric Order-flow Shock)
#
# ── adjpin() 실제 시그니처 (PINstimation 공식 문서 기준) ─────────────────────
#   adjpin(data, method = "ECM", initialsets = "GE", num_init = 20,
#          restricted = list(), ..., verbose = TRUE)
#
#   data        : {B, S} 열을 포함하는 data.frame (일별 매수·매도 건수)
#   method      : "ECM" (기본) 또는 "ML"
#   initialsets : "GE" | "CL" | "RANDOM" 또는 사용자 정의 data.frame
#                   ← data.frame 형식: 10열 (α, δ, θ, θ', εb, εs, μb, μs, Δb, Δs)
#                   ← 이전 윈도우 추정 파라미터를 직접 넘겨 warm-start 가능
#   num_init    : GE·CL·RANDOM 방식일 때 생성할 초기점 개수 (기본값 20)
#   verbose     : 진행 메시지 출력 여부
#
# ── 적응형 초기화 전략 (Adaptive Initialization) ────────────────────────────
#   60영업일 롤링 윈도우에서 연속 두 윈도우는 59일(98.3%)이 겹친다.
#   → 파라미터 공간이 거의 동일하므로 이전 추정값을 warm-start 초기점으로 활용.
#   → 전략:
#     · 첫 번째 윈도우 (i = WINDOW_SIZE):
#         initialsets = "GE",   num_init = NUM_INIT_FIRST (기본 20)
#         탐색 범위 넓게 → 전역 최적에 가까운 해 확보
#     · 이후 윈도우 (i > WINDOW_SIZE, 직전 추정 성공한 경우):
#         initialsets = 이전 파라미터 1행 data.frame (warm-start)
#         + num_init = NUM_INIT_ROLL (기본 5) 개의 GE 랜덤 초기점 추가
#         초기점 = 이전 수렴값 + 소수의 탐색 집합 → 수렴 안정성 유지
#     · 이후 윈도우 (직전 추정 실패한 경우):
#         warm-start 초기점 없으므로 첫 번째 윈도우와 동일하게 처리
#         initialsets = "GE",   num_init = NUM_INIT_FIRST
#
#   ※ PIN은 num_init 파라미터 자체가 없어 이 전략을 적용할 수 없다.
#     pin() 시그니처: pin(data, method, initialsets, verbose)
#     초기점 개수는 initialsets 종류에 따라 내부 고정 (예: "EA" → 5개)
#
# ─── 입출력 ───────────────────────────────────────────────────────────────────
# 입력  : R_output/{COUNTRY}/full_daily_bs.parquet  (01_preprocess.py 출력)
#           · PIN(00pin/)과 동일 파일 공유 — 먼저 실행된 쪽의 캐시 재사용
#           · 스키마: Symbol(Utf8), Date(Date), B(Int32), S(Int32)
#           · 영업일 캘린더 정렬 완료: 거래 없는 날에 B=S=0 행 삽입
# 출력  : R_output/{COUNTRY}/apin/
#           apin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet   ← 전체 추정 결과
#           apin_{COUNTRY}_{YYYYMMDD_HHMM}_sample1000.csv  ← 눈으로 확인용 1000행
#           checkpoints/sym_{Symbol}.parquet          ← 종목별 체크포인트
#
# ─── 처리 흐름 ────────────────────────────────────────────────────────────────
# [1] full_daily_bs.parquet 로드
#       · 영업일 캘린더 정렬이 완료된 파일이므로 60행 = 정확히 60 영업일
# [2] 체크포인트 확인
#       · checkpoints/sym_*.parquet 스캔 → 완료 종목 파악
#       · remaining_syms = setdiff(all_symbols, completed_syms)
#       · 이미 모두 완료됐으면 [3][4] 스킵 → [5] 병합으로 직행
# [3] 종목별 데이터 분할 + 워커 인자 구성
#       · split(daily_bs, Symbol) → 종목별 data.frame
#       · 각 종목 args 리스트 구성 (sym_df, 설정값 등)
# [4] 병렬 추정 — makeCluster(PSOCK) + parLapply
#       · PSOCK: Windows·Linux·macOS 모두 호환, 독립 Rscript 프로세스
#       · 1코어 = 1종목: 한 워커가 해당 종목의 전체 기간 롤링 추정 완주
#       · 윈도우(i − WINDOW_SIZE + 1 : i) 슬라이딩, i = WINDOW_SIZE … n_days
#         - valid_days < MIN_VALID_DAYS 인 윈도우 → 스킵
#         - adjpin() 수렴 실패 → 스킵 (warm-start 초기점 초기화)
#         - 첫 번째 윈도우 또는 직전 실패 → num_init = NUM_INIT_FIRST
#         - 직전 성공 → warm-start + num_init = NUM_INIT_ROLL
#       · 완료 즉시 checkpoints/sym_{Symbol}.parquet 독립 저장 (레이스컨디션 없음)
#       · CHECKPOINT_N 종목마다 진행률·ETA 로그 출력
# [5] 체크포인트 병합 → 최종 결과 저장
#       · 모든 sym_*.parquet → rbind → Symbol·Date 정렬
#       · parquet(전체) + CSV(샘플 1000행) 저장
#
# ─── 캘린더 정렬이 필요한 이유 ────────────────────────────────────────────────
#   틱에서 거래가 없는 날은 행 자체가 존재하지 않는다.
#   정렬 없이 60행 윈도우를 잡으면 실제로는 수개월치가 될 수 있어
#   추정 기준일이 부정확해진다. B=S=0 행 삽입 후 60행 = 정확히 60 영업일.
#
# ─── 실행 ─────────────────────────────────────────────────────────────────────
#   Rscript 01apin/02_r_apin.R
#
# ─── 의존 패키지 (최초 1회) ───────────────────────────────────────────────────
#   install.packages(c("PINstimation", "arrow", "parallel"))
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(arrow)
  library(parallel)
})


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR <- "E:/vpin_project_parquet/processing_data"
COUNTRY  <- "KOR"

# ── 롤링 윈도우 파라미터 ──────────────────────────────────────────────────────
WINDOW_SIZE    <- 60   # 롤링 윈도우 크기 (영업일)
MIN_VALID_DAYS <- 30   # 윈도우 내 실제 거래일(B+S>0) 최솟값

# ── adjpin() 파라미터 ─────────────────────────────────────────────────────────
# adjpin() 시그니처: adjpin(data, method, initialsets, num_init, restricted, ..., verbose)
ADJPIN_METHOD  <- "ECM"   # "ECM" (기본) 또는 "ML"
ADJPIN_INITSETS <- "GE"   # 초기점 생성 방법: "GE" | "CL" | "RANDOM"

# 적응형 num_init 설정
# · NUM_INIT_FIRST : 첫 번째 윈도우 또는 직전 추정 실패 시 사용 (넓은 탐색)
# · NUM_INIT_ROLL  : 직전 추정 성공 후 warm-start와 함께 사용 (좁은 탐색)
NUM_INIT_FIRST <- 20   # 첫 번째 윈도우 초기점 수
NUM_INIT_ROLL  <-  5   # 이후 롤링 윈도우 초기점 수 (warm-start 사용 시)

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
NUM_WORKERS  <- parallel::detectCores(logical = TRUE)
CHECKPOINT_N <- 100

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", COUNTRY, "full_daily_bs.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "apin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\n%s\n[APIN 전체계산] 시작: %s\n%s\n",
            strrep("=", 65), format(Sys.time(), "%Y-%m-%d %H:%M:%S"), strrep("=", 65)))
cat(sprintf("  나라코드    : %s\n", COUNTRY))
cat(sprintf("  INPUT_PATH  : %s\n", INPUT_PATH))
cat(sprintf("  OUTPUT_DIR  : %s\n", OUTPUT_DIR))
cat(sprintf("  윈도우 크기 : %d영업일  |  최소 유효일: %d일\n", WINDOW_SIZE, MIN_VALID_DAYS))
cat(sprintf("  Method      : %s  |  InitSets: %s\n", ADJPIN_METHOD, ADJPIN_INITSETS))
cat(sprintf("  num_init    : 첫 번째 윈도우 = %d  |  롤링 윈도우 = %d (warm-start)\n",
            NUM_INIT_FIRST, NUM_INIT_ROLL))
cat(sprintf("  CPU 코어    : %d개  |  로그 간격: %d종목\n", NUM_WORKERS, CHECKPOINT_N))


# =============================================================================
# [1] 데이터 로드
# =============================================================================

if (!file.exists(INPUT_PATH))
  stop(sprintf("[Error] 파일 없음: %s\n먼저 01_preprocess.py를 실행하세요.", INPUT_PATH))

cat(sprintf("\n[1] 데이터 로드: %s\n", INPUT_PATH))
daily_bs      <- arrow::read_parquet(INPUT_PATH)
daily_bs$Date <- as.Date(daily_bs$Date)
daily_bs$B    <- as.integer(daily_bs$B)
daily_bs$S    <- as.integer(daily_bs$S)

all_symbols <- sort(unique(daily_bs$Symbol))
n_total     <- length(all_symbols)

cat(sprintf("  전체 종목 수: %d개\n", n_total))
cat(sprintf("  전체 행     : %s\n", format(nrow(daily_bs), big.mark = ",")))
cat(sprintf("  날짜 범위   : %s ~ %s\n", min(daily_bs$Date), max(daily_bs$Date)))


# =============================================================================
# [2] 재개 확인
# =============================================================================

ckpt_files     <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$")
completed_syms <- gsub("^sym_|\\.parquet$", "", ckpt_files)
remaining_syms <- setdiff(all_symbols, completed_syms)

cat(sprintf("\n[2] 체크포인트 확인\n"))
cat(sprintf("  완료 종목  : %d개\n", length(completed_syms)))
cat(sprintf("  처리 예정  : %d개\n", length(remaining_syms)))

if (length(remaining_syms) == 0)
  cat("\n  모든 종목 처리 완료. 최종 병합으로 건너뜁니다.\n")


# =============================================================================
# [3] 종목 데이터 분할 + 워커 인자 구성
# =============================================================================

worker_process_symbol <- function(args) {
  suppressPackageStartupMessages({
    library(PINstimation)
    library(arrow)
  })

  sym_df         <- args$sym_df
  symbol         <- args$symbol
  checkpoint_dir <- args$checkpoint_dir
  window_size    <- args$window_size
  min_valid_days <- args$min_valid_days
  method         <- args$method
  initsets       <- args$initsets
  num_init_first <- args$num_init_first
  num_init_roll  <- args$num_init_roll

  n_days <- nrow(sym_df)

  # AdjPIN 파라미터 순서: (α, δ, θ, θ', εb, εs, μb, μs, Δb, Δs)
  # initialsets에 data.frame을 넘기면 해당 행을 초기점으로 직접 사용 (warm-start)
  # num_init는 "GE"/"CL"/"RANDOM" 방식일 때 생성할 초기점 개수를 제어
  # warm-start 시: 이전 파라미터 1행 data.frame + num_init_roll 개의 추가 GE 초기점
  #   → adjpin()이 initialsets(data.frame)과 num_init를 함께 받으면
  #     data.frame의 행을 포함해 총 초기점을 사용함
  make_warm_start_df <- function(prev_params) {
    data.frame(
      alpha  = prev_params["alpha"],
      delta  = prev_params["delta"],
      theta  = prev_params["theta"],
      thetap = prev_params["thetap"],
      eps.b  = prev_params["eps.b"],
      eps.s  = prev_params["eps.s"],
      mu.b   = prev_params["mu.b"],
      mu.s   = prev_params["mu.s"],
      d.b    = prev_params["d.b"],
      d.s    = prev_params["d.s"],
      stringsAsFactors = FALSE
    )
  }

  # 직전 윈도우 추정 파라미터 (warm-start 용)
  # NULL = 아직 성공한 추정 없음 → 첫 번째 윈도우 전략 사용
  prev_params <- NULL

  results <- list()

  if (n_days >= window_size) {
    for (i in window_size:n_days) {
      s          <- i - window_size + 1
      window_B   <- sym_df$B[s:i]
      window_S   <- sym_df$S[s:i]
      valid_days <- sum((window_B + window_S) > 0)
      if (valid_days < min_valid_days) next

      window_df <- data.frame(B = window_B, S = window_S)

      # 적응형 초기화:
      #   직전 추정 성공(prev_params 보유) → warm-start + num_init_roll
      #   직전 추정 실패 또는 첫 윈도우   → 표준 GE    + num_init_first
      if (!is.null(prev_params)) {
        cur_initialsets <- make_warm_start_df(prev_params)
        cur_num_init    <- num_init_roll
      } else {
        cur_initialsets <- initsets
        cur_num_init    <- num_init_first
      }

      est <- tryCatch({
        res    <- adjpin(data = window_df, method = method,
                         initialsets = cur_initialsets,
                         num_init    = cur_num_init,
                         verbose     = FALSE)
        params <- res@parameters
        list(a  = as.numeric(params["alpha"]),   d  = as.numeric(params["delta"]),
             t1 = as.numeric(params["theta"]),   t2 = as.numeric(params["thetap"]),
             ub = as.numeric(params["mu.b"]),    us = as.numeric(params["mu.s"]),
             eb = as.numeric(params["eps.b"]),   es = as.numeric(params["eps.s"]),
             pb = as.numeric(params["d.b"]),     ps = as.numeric(params["d.s"]),
             APIN = as.numeric(res@adjpin), PSOS = as.numeric(res@psos),
             params_raw = params,   # warm-start 전달용
             converged = TRUE)
      }, error = function(e) list(converged = FALSE))

      if (!isTRUE(est$converged)) {
        # 수렴 실패 → warm-start 초기화 (다음 윈도우에서 다시 넓은 탐색)
        prev_params <- NULL
        next
      }

      # 수렴 성공 → 다음 윈도우를 위해 파라미터 저장
      prev_params <- est$params_raw

      results[[length(results) + 1]] <- data.frame(
        Symbol = symbol, Date = as.Date(sym_df$Date[i]), valid_days = valid_days,
        a = est$a, d = est$d, t1 = est$t1, t2 = est$t2,
        ub = est$ub, us = est$us, eb = est$eb, es = est$es,
        pb = est$pb, ps = est$ps, APIN = est$APIN, PSOS = est$PSOS,
        stringsAsFactors = FALSE
      )
    }
  }

  result_df <- if (length(results) > 0) do.call(rbind, results) else
    data.frame(Symbol = character(0), Date = as.Date(character(0)),
               valid_days = integer(0),
               a = numeric(0), d = numeric(0), t1 = numeric(0), t2 = numeric(0),
               ub = numeric(0), us = numeric(0), eb = numeric(0), es = numeric(0),
               pb = numeric(0), ps = numeric(0), APIN = numeric(0), PSOS = numeric(0))

  out_path <- file.path(checkpoint_dir, paste0("sym_", symbol, ".parquet"))
  arrow::write_parquet(result_df, out_path, compression = "zstd")
  return(list(symbol = symbol, n_rows = nrow(result_df)))
}


if (length(remaining_syms) > 0) {

  cat(sprintf("\n[3] 종목 데이터 분할 (%d종목)...\n", length(remaining_syms)))

  split_bs <- split(daily_bs[, c("Date", "B", "S")], daily_bs$Symbol)

  symbol_args <- lapply(remaining_syms, function(sym) {
    rows <- split_bs[[sym]]
    rows <- rows[order(rows$Date), ]
    list(sym_df = rows, symbol = sym, checkpoint_dir = CHECKPOINT_DIR,
         window_size = WINDOW_SIZE, min_valid_days = MIN_VALID_DAYS,
         method = ADJPIN_METHOD, initsets = ADJPIN_INITSETS,
         num_init_first = NUM_INIT_FIRST, num_init_roll = NUM_INIT_ROLL)
  })
  cat("  분할 완료\n")

  # =============================================================================
  # [4] 병렬 처리
  # =============================================================================

  n_remaining <- length(remaining_syms)
  n_workers   <- min(NUM_WORKERS, n_remaining)

  cat(sprintf("\n[4] adjpin() 병렬 추정 — 워커 %d개, 윈도우 %d일\n%s\n",
              n_workers, WINDOW_SIZE, strrep("-", 65)))

  start_time <- Sys.time()
  cl <- makeCluster(n_workers, type = "PSOCK")
  on.exit(tryCatch(stopCluster(cl), error = function(e) NULL), add = TRUE)

  batch_starts <- seq(1, n_remaining, by = CHECKPOINT_N)

  for (bi in seq_along(batch_starts)) {
    b_start <- batch_starts[bi]
    b_end   <- min(b_start + CHECKPOINT_N - 1, n_remaining)
    tryCatch(
      parLapply(cl, symbol_args[b_start:b_end], worker_process_symbol),
      error = function(e) cat(sprintf("  [Warning] 배치 %d 오류: %s\n", bi, e$message))
    )
    n_done  <- length(completed_syms) + b_end
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    eta_min <- if (b_end > 0) elapsed / b_end * (n_remaining - b_end) else NA
    cat(sprintf("  [%s] %d / %d 종목 완료  (%.1f분 경과%s)\n",
                format(Sys.time(), "%H:%M:%S"), n_done, n_total, elapsed,
                if (!is.na(eta_min)) sprintf(" | 예상 잔여 %.0f분", eta_min) else ""))
  }

  cat(sprintf("\n  총 소요 시간: %.1f분\n",
              as.numeric(difftime(Sys.time(), start_time, units = "mins"))))
}


# =============================================================================
# [5] 체크포인트 병합 → 최종 결과 저장
# =============================================================================

cat(sprintf("\n[5] 체크포인트 병합 중...\n"))

all_ckpt_files <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$", full.names = TRUE)
cat(sprintf("  체크포인트 파일 수: %d개\n", length(all_ckpt_files)))

all_dfs   <- lapply(all_ckpt_files, function(f) tryCatch(arrow::read_parquet(f), error = function(e) NULL))
non_empty <- Filter(function(df) !is.null(df) && nrow(df) > 0, all_dfs)

if (length(non_empty) == 0) {
  cat("[Warning] 추정 결과가 없습니다.\n")
} else {
  final_df           <- do.call(rbind, non_empty)
  rownames(final_df) <- NULL
  final_df           <- final_df[order(final_df$Symbol, final_df$Date), ]

  run_id       <- format(Sys.time(), "%Y%m%d_%H%M")
  result_stem  <- sprintf("apin_%s_%s", COUNTRY, run_id)
  parquet_path <- file.path(OUTPUT_DIR, paste0(result_stem, ".parquet"))
  csv_path     <- file.path(OUTPUT_DIR, paste0(result_stem, "_sample1000.csv"))

  arrow::write_parquet(final_df, parquet_path, compression = "zstd")
  write.csv(head(final_df, 1000), csv_path, row.names = FALSE)

  cat(sprintf("\n%s\n[결과 요약]\n%s\n", strrep("=", 65), strrep("=", 65)))
  cat(sprintf("  나라코드    : %s\n", COUNTRY))
  cat(sprintf("  날짜 범위   : %s ~ %s\n", min(final_df$Date), max(final_df$Date)))
  cat(sprintf("  전체 레코드 : %s\n", format(nrow(final_df), big.mark = ",")))
  cat(sprintf("  완료 종목   : %d개 / 전체 %d개\n",
              length(unique(final_df$Symbol)), n_total))
  cat(sprintf("  APIN 범위   : %.4f ~ %.4f  (평균 %.4f)\n",
              min(final_df$APIN, na.rm = TRUE),
              max(final_df$APIN, na.rm = TRUE),
              mean(final_df$APIN, na.rm = TRUE)))
  cat(sprintf("\n  parquet (전체)     : %s\n", parquet_path))
  cat(sprintf("  CSV (샘플 1000행)  : %s\n", csv_path))
  cat(sprintf("  체크포인트         : %s\n", CHECKPOINT_DIR))
  cat(sprintf("%s\n", strrep("=", 65)))
}