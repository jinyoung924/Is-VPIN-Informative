# =============================================================================
# [VPIN 전체계산] PINstimation::vpin() 추정
# =============================================================================
#
# 모델: Easley et al.(2012) — Volume-Synchronized PIN (VPIN)
#   날짜 기반이 아닌 거래량 버킷 기반 추정.
#   ADV(일평균거래량) / BUCKETS_PER_DAY = 버킷 크기 VBS 를 자동 산출하고,
#   BVC(Bulk Volume Classification)로 버킷을 매수·매도로 분류한 뒤
#   SAMPLENGTH 개 버킷의 슬라이딩 평균으로 VPIN을 계산한다.
#   결과 단위는 날짜(Date)가 아닌 버킷(BucketNo)·버킷 종료 시각(agg_endtime).
#
# ─── 실제 vpin() 시그니처 (PINstimation 공식 문서 기준) ──────────────────────
#   vpin(data, timebarsize = 60, buckets = 50, samplength = 50,
#        tradinghours = 24, verbose = TRUE)
#
#   data        : {timestamp, price, volume} 순서의 data.frame (열 이름 불문, 순서 중요)
#   timebarsize : 타임바 크기 (단위: 초).  1분봉 = 60
#   buckets     : 일평균거래량 대비 버킷 수
#   samplength  : VPIN 계산 롤링 윈도우 버킷 수 (= window)
#   tradinghours: 하루 거래 시간 (단위: 시간).  한국 = 6.5
#   verbose     : 진행 상황 출력 여부
#
# ─── 버그 수정 내역 ────────────────────────────────────────────────────────────
# [수정 1] vpin() 파라미터 이름 오류 3개 수정
#   sessionlength → tradinghours  (패키지에 sessionlength 없음 → "unused argument" 에러)
#   window        → samplength    (패키지에 window 없음       → "unused argument" 에러)
#   timebarsize   : 단위가 '분'이 아닌 '초'. TIMEBARSIZE_MIN × 60 으로 변환 후 전달
#   → 위 3개 오류가 겹쳐 vpin() 호출이 항상 에러 → 결과 0건
#
# [수정 2] 결과 추출 슬롯 이름 오류 수정
#   result@data → result@bucketdata  (PINstimation S4 클래스의 실제 슬롯명)
#
# [수정 3] @bucketdata 타임스탬프 컬럼명 추가
#   @bucketdata의 실제 시각 컬럼은 "agg_endtime" / "agg_starttime"
#   기존 탐색 목록("timestamp","time","datetime")에 포함되지 않아 항상 NA 저장
#   → "agg_endtime", "agg_starttime" 추가
#
# [수정 4] vpin() 입력 data.frame 첫 번째 열 이름을 "timestamp"로 통일
#   vpin()은 열 이름 무관하게 순서(1=timestamp, 2=price, 3=volume)로 읽지만,
#   명시적으로 "timestamp"로 지정해 혼선 방지
#
# ─── 입출력 ───────────────────────────────────────────────────────────────────
# 입력  : R_output/{COUNTRY}/vpin/all_1m_bars.parquet  (01_preprocess.py 출력)
#           · 스키마: Symbol(Utf8), Datetime(Datetime), Price(Float64), Volume(Float64)
#           · 1분봉 집계 완료 (틱 → group_by_dynamic 1m)
# 출력  : R_output/{COUNTRY}/vpin/
#           vpin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet         ← 전체 추정 결과
#           vpin_{COUNTRY}_{YYYYMMDD_HHMM}_sample1000.csv  ← 눈으로 확인용 1000행
#           checkpoints/sym_{Symbol}.parquet               ← 종목별 체크포인트
#
# ─── 처리 흐름 ────────────────────────────────────────────────────────────────
# [1] all_1m_bars.parquet 로드
#       · Datetime → as.POSIXct(tz="Asia/Seoul") 변환
# [2] 체크포인트 확인
#       · checkpoints/sym_*.parquet 스캔 → 완료 종목 파악
#       · remaining_syms = setdiff(all_symbols, completed_syms)
# [3] 종목별 데이터 분할 + 워커 인자 구성
#       · split(bars_df, Symbol) → 종목별 data.frame
#       · 각 종목 args 리스트: sym_df, tradinghours, buckets_per_day, 등
# [4] 병렬 추정 — makeCluster(PSOCK) + parLapply
#       · 워커: vpin(data = ..., timebarsize = TIMEBARSIZE_MIN * 60,
#                    buckets = BUCKETS_PER_DAY, samplength = SAMPLENGTH,
#                    tradinghours = TRADINGHOURS, verbose = FALSE)
#       · S4 결과 객체에서 @bucketdata (버킷 단위 DataFrame) 추출
#         - @bucketdata 타임스탬프 컬럼: "agg_endtime" 우선 탐색
#         - @bucketdata 실패 시 @vpin 벡터 사용 (Datetime=NA)
#       · 완료 즉시 checkpoints/sym_{Symbol}.parquet 독립 저장
# [5] 체크포인트 병합 → 최종 결과 저장
#       · Arrow open_dataset + dplyr::compute() 방식 (메모리 효율)
#         - open_dataset: lazy 참조 (C++ 메모리, R 힙 미사용)
#         - arrange + compute(): Arrow C++ 에서 정렬 후 Arrow Table 반환
#         - write_parquet: Arrow Table을 스트리밍으로 파일에 씀
#       · parquet(전체) + CSV(샘플 1000행) 저장
#
# ─── VPIN이 OOM 위험을 가지는 이유 ──────────────────────────────────────────
#   날짜 기반 PIN/APIN은 4,000종목 × 400영업일 ≈ 152만 행이지만,
#   VPIN은 버킷 기반으로 종목당 수천~수만 버킷 → 전체 수천만 행이 될 수 있다.
#   lapply + rbind는 전체 데이터 복사본을 생성해 피크 메모리가 2배에 달하므로
#   Arrow 스트리밍 병합을 사용한다.
#
# ─── 실행 ─────────────────────────────────────────────────────────────────────
#   Rscript 02vpin/02_r_vpin.R
#
# ─── 의존 패키지 (최초 1회) ───────────────────────────────────────────────────
#   install.packages(c("PINstimation", "arrow", "dplyr", "parallel"))
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(arrow)
  library(dplyr)     # open_dataset 스트리밍 집계에 필요
  library(parallel)
})


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR <- "E:/vpin_project_parquet/processing_data"
COUNTRY  <- "KOR"

# ── vpin() 파라미터 ─────────────────────────────────────────────────────────
# vpin() 시그니처: vpin(data, timebarsize, buckets, samplength, tradinghours, verbose)
#
SAMPLENGTH      <- 50    # 롤링 윈도우 버킷 수 (vpin() 의 samplength 인자)
BUCKETS_PER_DAY <- 50    # 일평균거래량 대비 버킷 수 (vpin() 의 buckets 인자)
TRADINGHOURS    <- 6.5   # 하루 거래 시간 시간(hour) 단위  (vpin() 의 tradinghours 인자)
                         #   한국 KRX: 09:00-15:30 = 6.5h
TIMEBARSIZE_MIN <- 1     # 타임바 크기 (분 단위, 사람이 읽기 쉬운 형식)
                         #   vpin() 에 전달 시 × 60 하여 초(second) 단위로 변환

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
NUM_WORKERS  <- parallel::detectCores(logical = TRUE)
CHECKPOINT_N <- 100

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", COUNTRY, "vpin", "all_1m_bars.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "vpin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\n%s\n[VPIN 전체계산] 시작: %s\n%s\n",
            strrep("=", 65), format(Sys.time(), "%Y-%m-%d %H:%M:%S"), strrep("=", 65)))
cat(sprintf("  나라코드       : %s\n", COUNTRY))
cat(sprintf("  INPUT_PATH     : %s\n", INPUT_PATH))
cat(sprintf("  OUTPUT_DIR     : %s\n", OUTPUT_DIR))
cat(sprintf("  samplength     : %d 버킷  |  buckets/day: %d  |  tradinghours: %.1fh\n",
            SAMPLENGTH, BUCKETS_PER_DAY, TRADINGHOURS))
cat(sprintf("  timebarsize    : %d분 (%d초)\n", TIMEBARSIZE_MIN, TIMEBARSIZE_MIN * 60L))
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

  # [수정 4] 첫 번째 열을 "timestamp"로 명시
  # vpin()은 열 순서({timestamp, price, volume})로 읽으므로
  # 열 이름을 맞춰 주면 혼선 없음
  vpin_input <- data.frame(
    timestamp = as.POSIXct(sym_df$Datetime, tz = "Asia/Seoul"),
    price     = as.numeric(sym_df$Price),
    volume    = as.numeric(sym_df$Volume),
    stringsAsFactors = FALSE
  )

  # [수정 1] 올바른 파라미터 이름으로 vpin() 호출
  #   sessionlength → tradinghours
  #   window        → samplength
  #   timebarsize   → 분 단위가 아닌 초 단위 (× 60)
  result <- tryCatch({
    vpin(data         = vpin_input,
         timebarsize  = args$timebarsize_sec,   # 초 단위 (1분봉 = 60초)
         buckets      = args$buckets_per_day,
         samplength   = args$samplength,        # 롤링 윈도우 버킷 수
         tradinghours = args$tradinghours,       # 하루 거래 시간 (시간 단위)
         verbose      = FALSE)
  }, error = function(e) NULL)

  result_df <- tryCatch({
    if (is.null(result)) stop("vpin() 반환값 없음")

    # [수정 2] @data → @bucketdata  (PINstimation S4 슬롯의 실제 이름)
    bucket_data <- tryCatch(result@bucketdata, error = function(e) NULL)

    if (!is.null(bucket_data) && nrow(bucket_data) > 0) {
      names(bucket_data) <- tolower(names(bucket_data))

      # [수정 3] @bucketdata의 실제 타임스탬프 컬럼명 추가
      # PINstimation @bucketdata 실제 컬럼: agg_endtime, agg_starttime 등
      ts_col   <- intersect(
        c("agg_endtime", "agg_starttime", "timestamp", "time", "datetime"),
        names(bucket_data)
      )[1]
      vpin_col <- intersect(c("vpin", "adjvpin"), names(bucket_data))[1]

      if (is.na(ts_col))   ts_col   <- names(bucket_data)[1]
      if (is.na(vpin_col)) vpin_col <- names(bucket_data)[2]

      data.frame(
        Symbol   = symbol,
        Datetime = as.POSIXct(bucket_data[[ts_col]], tz = "Asia/Seoul"),
        BucketNo = seq_len(nrow(bucket_data)),
        VPIN     = as.numeric(bucket_data[[vpin_col]]),
        stringsAsFactors = FALSE
      )
    } else {
      vpin_vec <- result@vpin
      if (length(vpin_vec) == 0) stop("vpin 벡터 없음")
      data.frame(
        Symbol   = symbol,
        Datetime = as.POSIXct(NA),
        BucketNo = seq_along(vpin_vec),
        VPIN     = as.numeric(vpin_vec),
        stringsAsFactors = FALSE
      )
    }
  }, error = function(e) {
    data.frame(
      Symbol   = character(0),
      Datetime = as.POSIXct(character(0)),
      BucketNo = integer(0),
      VPIN     = numeric(0)
    )
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
    list(
      sym_df          = rows,
      symbol          = sym,
      checkpoint_dir  = CHECKPOINT_DIR,
      timebarsize_sec = TIMEBARSIZE_MIN * 60L,  # 초 단위로 변환
      buckets_per_day = BUCKETS_PER_DAY,
      samplength      = SAMPLENGTH,
      tradinghours    = TRADINGHOURS
    )
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
#
# VPIN은 버킷 기반으로 수천만 행이 될 수 있다.
# lapply + do.call(rbind) 방식은 rbind 시 전체 데이터 복사본을 생성해
# 피크 메모리가 실제 데이터 크기의 2배에 달하므로 OOM 위험이 있다.
#
# → Arrow open_dataset + dplyr::compute() 방식으로 대체:
#     open_dataset : 파일 목록을 지연(lazy) 참조 — 아직 메모리 로드 없음
#     dplyr::arrange + compute() : Arrow C++ 메모리에서 정렬 후 Arrow Table 반환
#                                  R 힙을 거치지 않아 메모리 약 3배 절감
#     write_parquet : Arrow Table을 스트리밍으로 파일에 씀
#     집계 통계     : 별도 lazy 집계로 계산 — 전체 데이터 R로 가져오지 않음
#     샘플 CSV      : head(1000) + collect() 로 1000행만 R로 가져와 저장
# =============================================================================

cat(sprintf("\n[5] 체크포인트 병합 중...\n"))

all_ckpt_files <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$", full.names = TRUE)
cat(sprintf("  체크포인트 파일 수: %d개\n", length(all_ckpt_files)))

if (length(all_ckpt_files) == 0) {
  cat("[Warning] 추정 결과가 없습니다.\n")
} else {

  ckpt_ds <- tryCatch(
    arrow::open_dataset(CHECKPOINT_DIR, format = "parquet"),
    error = function(e) { cat(sprintf("[Error] 데이터셋 열기 실패: %s\n", e$message)); NULL }
  )

  if (is.null(ckpt_ds)) {
    cat("[Warning] 추정 결과를 열 수 없습니다.\n")
  } else {

    stats <- tryCatch(
      ckpt_ds |>
        summarise(
          n_rows    = n(),
          n_syms    = n_distinct(Symbol),
          vpin_min  = min(VPIN, na.rm = TRUE),
          vpin_max  = max(VPIN, na.rm = TRUE),
          vpin_mean = mean(VPIN, na.rm = TRUE)
        ) |>
        collect(),
      error = function(e) NULL
    )

    if (is.null(stats) || stats$n_rows == 0) {
      cat("[Warning] 추정 결과가 없습니다.\n")
    } else {

      run_id       <- format(Sys.time(), "%Y%m%d_%H%M")
      result_stem  <- sprintf("vpin_%s_%s", COUNTRY, run_id)
      parquet_path <- file.path(OUTPUT_DIR, paste0(result_stem, ".parquet"))
      csv_path     <- file.path(OUTPUT_DIR, paste0(result_stem, "_sample1000.csv"))

      cat(sprintf("  총 버킷 수 : %s행\n", format(stats$n_rows, big.mark = ",")))
      cat(sprintf("  Arrow 스트리밍 병합 시작...\n"))

      result_tbl <- ckpt_ds |>
        arrange(Symbol, Datetime) |>
        compute()

      arrow::write_parquet(result_tbl, parquet_path, compression = "zstd")
      rm(result_tbl); gc()

      sample_df <- ckpt_ds |>
        arrange(Symbol, Datetime) |>
        head(1000) |>
        collect()
      write.csv(sample_df, csv_path, row.names = FALSE)
      rm(sample_df)

      cat(sprintf("\n%s\n[결과 요약]\n%s\n", strrep("=", 65), strrep("=", 65)))
      cat(sprintf("  나라코드    : %s\n", COUNTRY))
      cat(sprintf("  전체 버킷 수: %s\n", format(stats$n_rows, big.mark = ",")))
      cat(sprintf("  완료 종목   : %d개 / 전체 %d개\n", stats$n_syms, n_total))
      cat(sprintf("  VPIN 범위   : %.6f ~ %.6f  (평균 %.6f)\n",
                  stats$vpin_min, stats$vpin_max, stats$vpin_mean))
      cat(sprintf("\n  parquet (전체)     : %s\n", parquet_path))
      cat(sprintf("  CSV (샘플 1000행)  : %s\n", csv_path))
      cat(sprintf("  체크포인트         : %s\n", CHECKPOINT_DIR))
      cat(sprintf("%s\n", strrep("=", 65)))
    }
  }
}