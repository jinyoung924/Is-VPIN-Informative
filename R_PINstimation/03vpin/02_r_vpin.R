# =============================================================================
# [VPIN 전체계산] PINstimation::vpin() 추정
# =============================================================================
#
# 모델: Easley et al.(2012) — Volume-Synchronized PIN
#
# 입력  : R_output/{DATA_FOLDER}/vpin/all_1m_bars.parquet  (01_preprocess.py 출력)
# 출력  : R_output/{DATA_FOLDER}/vpin/
#           r_vpin_{DATA_FOLDER}_{YYYYMMDD_HHMM}.parquet  ← 나라·기간·완료일시 포함
#           r_vpin_{DATA_FOLDER}_{YYYYMMDD_HHMM}.csv
#           checkpoints/sym_{Symbol}.parquet
#
# 실행:
#   Rscript 03vpin/02_r_vpin.R
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

ROLLING_WINDOW  <- 50
BUCKETS_PER_DAY <- 50
SESSION_LENGTH  <- 6.5   # 한국: 09:00-15:30 = 6.5시간
TIMEBARSIZE     <- 1     # 1분봉

NUM_WORKERS  <- parallel::detectCores(logical = TRUE)
CHECKPOINT_N <- 100

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", DATA_FOLDER, "vpin", "all_1m_bars.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", DATA_FOLDER, "vpin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

folder_parts <- strsplit(DATA_FOLDER, "_")[[1]]
COUNTRY      <- folder_parts[1]
PERIOD       <- paste(folder_parts[-1], collapse = "_")

cat(sprintf("\n%s\n[VPIN 전체계산] 시작: %s\n%s\n",
            strrep("=", 65), format(Sys.time(), "%Y-%m-%d %H:%M:%S"), strrep("=", 65)))
cat(sprintf("  데이터         : %s  (나라: %s, 기간: %s)\n", DATA_FOLDER, COUNTRY, PERIOD))
cat(sprintf("  INPUT_PATH     : %s\n", INPUT_PATH))
cat(sprintf("  OUTPUT_DIR     : %s\n", OUTPUT_DIR))
cat(sprintf("  롤링 윈도우    : %d 버킷  |  하루 버킷: %d  |  세션: %.1f시간\n",
            ROLLING_WINDOW, BUCKETS_PER_DAY, SESSION_LENGTH))
cat(sprintf("  CPU 코어       : %d개  |  로그 간격: %d종목\n", NUM_WORKERS, CHECKPOINT_N))


# =============================================================================
# [1] 데이터 로드
# =============================================================================

if (!file.exists(INPUT_PATH))
  stop(sprintf("[Error] 파일 없음: %s\n먼저 01_preprocess.py를 실행하세요.", INPUT_PATH))

cat(sprintf("\n[1] 데이터 로드: %s\n", INPUT_PATH))
bars_df          <- arrow::read_parquet(INPUT_PATH)
bars_df$Datetime <- as.POSIXct(bars_df$Datetime, tz = "Asia/Seoul")

all_symbols <- sort(unique(bars_df$Symbol))
n_total     <- length(all_symbols)

cat(sprintf("  전체 종목 수: %d개\n", n_total))
cat(sprintf("  전체 행     : %s\n", format(nrow(bars_df), big.mark = ",")))
cat(sprintf("  시간 범위   : %s ~ %s\n", min(bars_df$Datetime), max(bars_df$Datetime)))


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

  vpin_input <- data.frame(
    Datetime = as.POSIXct(sym_df$Datetime, tz = "Asia/Seoul"),
    price    = as.numeric(sym_df$Price),
    volume   = as.numeric(sym_df$Volume),
    stringsAsFactors = FALSE
  )

  result <- tryCatch({
    vpin(data = vpin_input, sessionlength = args$session_length,
         buckets = args$buckets_per_day, window = args$rolling_window,
         timebarsize = args$timebarsize, verbose = FALSE)
  }, error = function(e) NULL)

  result_df <- tryCatch({
    if (is.null(result)) stop("vpin() 반환값 없음")

    bucket_data <- tryCatch(result@data, error = function(e) NULL)

    if (!is.null(bucket_data) && nrow(bucket_data) > 0) {
      names(bucket_data) <- tolower(names(bucket_data))
      ts_col   <- intersect(c("timestamp", "time", "datetime"), names(bucket_data))[1]
      vpin_col <- intersect(c("vpin", "adjvpin"),               names(bucket_data))[1]
      if (is.na(ts_col))   ts_col   <- names(bucket_data)[1]
      if (is.na(vpin_col)) vpin_col <- names(bucket_data)[2]

      data.frame(Symbol = symbol,
                 Datetime = as.POSIXct(bucket_data[[ts_col]], tz = "Asia/Seoul"),
                 BucketNo = seq_len(nrow(bucket_data)),
                 VPIN = as.numeric(bucket_data[[vpin_col]]),
                 stringsAsFactors = FALSE)
    } else {
      vpin_vec <- result@vpin
      if (length(vpin_vec) == 0) stop("vpin 벡터 없음")
      data.frame(Symbol = symbol, Datetime = as.POSIXct(NA),
                 BucketNo = seq_along(vpin_vec),
                 VPIN = as.numeric(vpin_vec), stringsAsFactors = FALSE)
    }
  }, error = function(e) {
    data.frame(Symbol = character(0), Datetime = as.POSIXct(character(0)),
               BucketNo = integer(0), VPIN = numeric(0))
  })

  out_path <- file.path(checkpoint_dir, paste0("sym_", symbol, ".parquet"))
  arrow::write_parquet(result_df, out_path, compression = "zstd")
  return(list(symbol = symbol, n_rows = nrow(result_df)))
}


if (length(remaining_syms) > 0) {

  cat(sprintf("\n[3] 종목 데이터 분할 (%d종목)...\n", length(remaining_syms)))

  split_bars <- split(bars_df[, c("Datetime", "Price", "Volume")], bars_df$Symbol)

  symbol_args <- lapply(remaining_syms, function(sym) {
    rows <- split_bars[[sym]]
    rows <- rows[order(rows$Datetime), ]
    list(sym_df = rows, symbol = sym, checkpoint_dir = CHECKPOINT_DIR,
         session_length = SESSION_LENGTH, buckets_per_day = BUCKETS_PER_DAY,
         rolling_window = ROLLING_WINDOW, timebarsize = TIMEBARSIZE)
  })
  cat("  분할 완료\n")

  # =============================================================================
  # [4] 병렬 처리
  # =============================================================================

  n_remaining <- length(remaining_syms)
  n_workers   <- min(NUM_WORKERS, n_remaining)

  cat(sprintf("\n[4] vpin() 병렬 추정 — 워커 %d개\n%s\n",
              n_workers, strrep("-", 65)))

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
  final_df           <- final_df[order(final_df$Symbol, final_df$Datetime), ]

  run_id       <- format(Sys.time(), "%Y%m%d_%H%M")
  result_stem  <- sprintf("r_vpin_%s_%s", DATA_FOLDER, run_id)
  csv_path     <- file.path(OUTPUT_DIR, paste0(result_stem, ".csv"))
  parquet_path <- file.path(OUTPUT_DIR, paste0(result_stem, ".parquet"))

  write.csv(final_df, csv_path, row.names = FALSE)
  arrow::write_parquet(final_df, parquet_path, compression = "zstd")

  vpin_ok <- final_df$VPIN[!is.na(final_df$VPIN)]
  cat(sprintf("\n%s\n[결과 요약]\n%s\n", strrep("=", 65), strrep("=", 65)))
  cat(sprintf("  데이터      : %s  (나라: %s, 기간: %s)\n", DATA_FOLDER, COUNTRY, PERIOD))
  cat(sprintf("  전체 버킷 수: %s\n", format(nrow(final_df), big.mark = ",")))
  cat(sprintf("  완료 종목   : %d개 / 전체 %d개\n",
              length(unique(final_df$Symbol)), n_total))
  if (length(vpin_ok) > 0)
    cat(sprintf("  VPIN 범위   : %.6f ~ %.6f  (평균 %.6f)\n",
                min(vpin_ok), max(vpin_ok), mean(vpin_ok)))
  cat(sprintf("\n  CSV     : %s\n", csv_path))
  cat(sprintf("  parquet : %s\n", parquet_path))
  cat(sprintf("  체크포인트: %s\n", CHECKPOINT_DIR))
  cat(sprintf("%s\n", strrep("=", 65)))
}
