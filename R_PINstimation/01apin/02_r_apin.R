# =============================================================================
# [APIN 전체계산] PINstimation::adjpin() 롤링 추정
# =============================================================================
#
# 모델: Duarte & Young(2009) — 10개 파라미터
#
# 입력  : R_output/{DATA_FOLDER}/full_daily_bs.parquet  (01_preprocess.py 출력)
#           PIN(00pin/)과 동일한 파일을 공유한다.
# 출력  : R_output/{DATA_FOLDER}/apin/
#           r_apin_{DATA_FOLDER}_{YYYYMMDD_HHMM}.parquet  ← 나라·기간·완료일시 포함
#           r_apin_{DATA_FOLDER}_{YYYYMMDD_HHMM}.csv
#           checkpoints/sym_{Symbol}.parquet
#
# 실행:
#   Rscript 01apin/02_r_apin.R
#
# 의존 패키지 (최초 1회):
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

BASE_DIR    <- "E:/vpin_project_parquet"
DATA_FOLDER <- "KOR_201910_202107"

WINDOW_SIZE    <- 60
MIN_VALID_DAYS <- 30

ADJPIN_METHOD      <- "ECM"
ADJPIN_INITIALSETS <- "GE"
NUM_INITIAL_SETS   <- 20

NUM_WORKERS  <- parallel::detectCores(logical = TRUE)
CHECKPOINT_N <- 100

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", DATA_FOLDER, "full_daily_bs.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", DATA_FOLDER, "apin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

folder_parts <- strsplit(DATA_FOLDER, "_")[[1]]
COUNTRY      <- folder_parts[1]
PERIOD       <- paste(folder_parts[-1], collapse = "_")

cat(sprintf("\n%s\n[APIN 전체계산] 시작: %s\n%s\n",
            strrep("=", 65), format(Sys.time(), "%Y-%m-%d %H:%M:%S"), strrep("=", 65)))
cat(sprintf("  데이터      : %s  (나라: %s, 기간: %s)\n", DATA_FOLDER, COUNTRY, PERIOD))
cat(sprintf("  INPUT_PATH  : %s\n", INPUT_PATH))
cat(sprintf("  OUTPUT_DIR  : %s\n", OUTPUT_DIR))
cat(sprintf("  윈도우 크기 : %d영업일  |  최소 유효일: %d일\n", WINDOW_SIZE, MIN_VALID_DAYS))
cat(sprintf("  Method      : %s  |  InitSets: %s (%d세트)\n",
            ADJPIN_METHOD, ADJPIN_INITIALSETS, NUM_INITIAL_SETS))
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
  initialsets    <- args$initialsets
  num_init       <- args$num_init

  n_days <- nrow(sym_df)

  estimate_window <- function(window_df) {
    tryCatch({
      res    <- adjpin(data = window_df, method = method,
                       initialsets = initialsets, num_init = num_init,
                       verbose = FALSE)
      params <- res@parameters
      list(a  = as.numeric(params["alpha"]),   d  = as.numeric(params["delta"]),
           t1 = as.numeric(params["theta"]),   t2 = as.numeric(params["thetap"]),
           ub = as.numeric(params["mu.b"]),    us = as.numeric(params["mu.s"]),
           eb = as.numeric(params["eps.b"]),   es = as.numeric(params["eps.s"]),
           pb = as.numeric(params["d.b"]),     ps = as.numeric(params["d.s"]),
           APIN = as.numeric(res@adjpin), PSOS = as.numeric(res@psos),
           converged = TRUE)
    }, error = function(e) list(converged = FALSE))
  }

  results <- list()
  if (n_days >= window_size) {
    for (i in window_size:n_days) {
      s          <- i - window_size + 1
      window_B   <- sym_df$B[s:i]
      window_S   <- sym_df$S[s:i]
      valid_days <- sum((window_B + window_S) > 0)
      if (valid_days < min_valid_days) next

      est <- estimate_window(data.frame(B = window_B, S = window_S))
      if (!isTRUE(est$converged)) next

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
         method = ADJPIN_METHOD, initialsets = ADJPIN_INITIALSETS,
         num_init = NUM_INITIAL_SETS)
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
  result_stem  <- sprintf("r_apin_%s_%s", DATA_FOLDER, run_id)
  csv_path     <- file.path(OUTPUT_DIR, paste0(result_stem, ".csv"))
  parquet_path <- file.path(OUTPUT_DIR, paste0(result_stem, ".parquet"))

  write.csv(final_df, csv_path, row.names = FALSE)
  arrow::write_parquet(final_df, parquet_path, compression = "zstd")

  cat(sprintf("\n%s\n[결과 요약]\n%s\n", strrep("=", 65), strrep("=", 65)))
  cat(sprintf("  데이터      : %s  (나라: %s, 기간: %s)\n", DATA_FOLDER, COUNTRY, PERIOD))
  cat(sprintf("  전체 레코드 : %s\n", format(nrow(final_df), big.mark = ",")))
  cat(sprintf("  완료 종목   : %d개 / 전체 %d개\n",
              length(unique(final_df$Symbol)), n_total))
  cat(sprintf("  APIN 범위   : %.4f ~ %.4f  (평균 %.4f)\n",
              min(final_df$APIN, na.rm = TRUE),
              max(final_df$APIN, na.rm = TRUE),
              mean(final_df$APIN, na.rm = TRUE)))
  cat(sprintf("\n  CSV     : %s\n", csv_path))
  cat(sprintf("  parquet : %s\n", parquet_path))
  cat(sprintf("  체크포인트: %s\n", CHECKPOINT_DIR))
  cat(sprintf("%s\n", strrep("=", 65)))
}
