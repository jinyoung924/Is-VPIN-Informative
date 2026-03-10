"""
=============================================================================
[APIN 파이프라인 - Step 1+2] Python 전처리
틱 데이터 (월별 parquet)
  → Step 1: 일별 B/S 집계     → all_daily_bs.parquet   (PIN·APIN 공유 캐시)
  → Step 2: 영업일 캘린더 정렬 → full_daily_bs.parquet  (R 입력)
=============================================================================

PIN(00pin/)과 all_daily_bs.parquet·full_daily_bs.parquet 캐시를 공유한다.
어느 쪽을 먼저 실행해도 기존 캐시를 자동으로 재사용한다.

지원 입력 폴더 형식: {나라코드}_{시작연도월}_{종료연도월}
  예) KOR_201910_202107  /  US_201901_202012  /  JP_202001_202312

지원 나라코드: KOR  US  JP  CA  FR  GR  HK  IT  UK

실행 순서:
  1) python 01apin/01_preprocess.py
  2) Rscript 01apin/02_r_apin.R

=============================================================================
"""

import os
import re
import sys
import glob
import math
import polars as pl
import pyarrow.parquet as pq
import warnings
from datetime import datetime
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

# 틱 parquet 루트 폴더
BASE_DIR = r"E:\vpin_project_parquet"

# 처리할 데이터 폴더명 (BASE_DIR 하위)
# 형식: {나라코드}_{시작YYYYMM}_{종료YYYYMM}
DATA_FOLDER = "KOR_201910_202107"

# True → 기존 캐시 파일이 있어도 강제 재생성
FORCE_REPROCESS = False

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

VALID_COUNTRIES = {"KOR", "US", "JP", "CA", "FR", "GR", "HK", "IT", "UK"}
_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{6})_(\d{6})$")

def _parse_folder(folder: str):
    m = _FOLDER_RE.match(folder)
    if not m or m.group(1) not in VALID_COUNTRIES:
        raise ValueError(
            f"DATA_FOLDER 형식 오류: '{folder}'\n"
            f"  올바른 형식: {{나라코드}}_{{YYYYMM}}_{{YYYYMM}}\n"
            f"  예) KOR_201910_202107\n"
            f"  지원 나라코드: {', '.join(sorted(VALID_COUNTRIES))}"
        )
    return m.group(1), m.group(2), m.group(3)

COUNTRY, PERIOD_START, PERIOD_END = _parse_folder(DATA_FOLDER)

DATA_DIR  = os.path.join(BASE_DIR, DATA_FOLDER)
CACHE_DIR = os.path.join(BASE_DIR, "R_output", DATA_FOLDER)


# =============================================================================
# Step 1: 틱 데이터 → all_daily_bs.parquet
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)
    df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "LR"])
        .filter(pl.col("LR").is_in([1, -1]))
        .with_columns([
            (pl.col("LR") == 1).cast(pl.UInt32).alias("is_Buy"),
            (pl.col("LR") == -1).cast(pl.UInt32).alias("is_Sell"),
        ])
        .group_by(["Symbol", "Date"])
        .agg([pl.col("is_Buy").sum().alias("B"), pl.col("is_Sell").sum().alias("S")])
        .filter((pl.col("B") > 0) | (pl.col("S") > 0))
        .sort(["Symbol", "Date"])
        .collect()
    )
    if df["Date"].dtype != pl.Date:
        df = df.with_columns(pl.col("Date").cast(pl.Date))
    print(f"{df.height:,} records")
    return df


def get_parquet_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"\n[파일 탐색] {data_dir}")
    print(f"  parquet {len(files)}개 발견")
    return files


def run_step1(data_dir: str, cache_dir: str) -> pl.DataFrame:
    """틱 parquet 전체를 순회하여 all_daily_bs.parquet로 집계한다."""
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.join(cache_dir, "all_daily_bs.parquet")

    if not FORCE_REPROCESS and os.path.exists(output_path):
        print(f"\n[Step 1 스킵] 기존 캐시 재사용: {output_path}")
        print(f"  (00pin/01_preprocess.py 가 생성한 공유 캐시)")
        return pl.read_parquet(output_path)

    parquet_files = get_parquet_files(data_dir)
    if not parquet_files:
        raise RuntimeError(f"[Error] parquet 파일을 찾을 수 없습니다: {data_dir}")

    print(f"\n{'='*65}")
    print(f"[Step 1] 틱 데이터 → 일별 B/S 집계  [{DATA_FOLDER}]")
    print(f"{'='*65}\n")

    all_dfs = []
    for path in parquet_files:
        df = preprocess_trade_data_polars(path)
        if not df.is_empty():
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError("[Error] 유효한 데이터가 없습니다.")

    full_df = (
        pl.concat(all_dfs, how="vertical")
        .group_by(["Symbol", "Date"])
        .agg([pl.col("B").sum(), pl.col("S").sum()])
        .sort(["Symbol", "Date"])
    )
    full_df.write_parquet(output_path, compression="zstd")

    print(f"\n[Step 1 완료] {output_path}")
    print(f"  행 수  : {full_df.height:,} | 종목 수: {full_df['Symbol'].n_unique():,}")
    print(f"  기간   : {full_df['Date'].min()} ~ {full_df['Date'].max()}")
    return full_df


# =============================================================================
# Step 2: 영업일 캘린더 정렬 → full_daily_bs.parquet
# =============================================================================

def run_step2(daily_bs: pl.DataFrame, cache_dir: str) -> str:
    """
    영업일 캘린더 정렬: 거래 없는 날에 B=S=0 행을 삽입한다.

    전략: 청크 cross join + PyArrow 증분 쓰기
      - CHUNK_SIZE 종목씩 cross join → Polars 벡터 연산으로 속도 유지
      - 청크 처리 후 메모리 즉시 해제 → 피크 메모리 최소화
      - 결과를 누적하지 않고 파일에 바로 씀 → 메모리 스파이크 없음

    메모리 추정 (CHUNK_SIZE=500, 2500영업일 기준):
      청크당 피크 ≈ 500 × 2500 × 20 bytes ≈ 25 MB
    """
    CHUNK_SIZE = 500

    output_path = os.path.join(cache_dir, "full_daily_bs.parquet")

    if FORCE_REPROCESS and os.path.exists(output_path):
        os.remove(output_path)

    if os.path.exists(output_path):
        print(f"\n[Step 2 스킵] 기존 캐시 재사용: {output_path}")
        return output_path

    print(f"\n{'='*65}")
    print(f"[Step 2] 영업일 캘린더 정렬  [{DATA_FOLDER}]")
    print(f"{'='*65}")

    all_symbols = daily_bs["Symbol"].unique().sort().to_list()
    calendar_df = daily_bs.select("Date").unique().sort("Date")
    n_sym       = len(all_symbols)
    n_days      = len(calendar_df)
    n_chunks    = math.ceil(n_sym / CHUNK_SIZE)
    chunk_mb    = max(1, CHUNK_SIZE * n_days * 20 // (1024 * 1024))

    print(f"  종목 수    : {n_sym:,}개  →  {n_chunks}개 청크 (청크당 최대 {CHUNK_SIZE}종목)")
    print(f"  영업일 수  : {n_days:,}일")
    print(f"  예상 총 행 : {n_sym * n_days:,}행")
    print(f"  청크당 메모리: ~{chunk_mb}MB\n")

    writer = None
    try:
        for chunk_idx in range(n_chunks):
            chunk_syms = all_symbols[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]

            chunk_bs      = daily_bs.filter(pl.col("Symbol").is_in(chunk_syms))
            chunk_syms_df = pl.DataFrame({"Symbol": chunk_syms})
            grid          = chunk_syms_df.join(calendar_df, how="cross")

            aligned = (
                grid
                .join(chunk_bs, on=["Symbol", "Date"], how="left")
                .with_columns([
                    pl.col("B").fill_null(0).cast(pl.Int32),
                    pl.col("S").fill_null(0).cast(pl.Int32),
                ])
                .sort(["Symbol", "Date"])
            )

            table = aligned.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)

            del aligned, table, grid, chunk_bs, chunk_syms_df

            n_done = min((chunk_idx + 1) * CHUNK_SIZE, n_sym)
            print(f"  청크 {chunk_idx + 1}/{n_chunks} 완료  ({n_done}/{n_sym}종목)")

    finally:
        if writer:
            writer.close()

    print(f"\n[Step 2 완료] {output_path}")
    print(f"  총 행 수: {n_sym * n_days:,}  ({n_sym}종목 × {n_days}영업일)")
    return output_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[APIN 전처리] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  데이터   : {DATA_FOLDER}  (나라: {COUNTRY}, 기간: {PERIOD_START}~{PERIOD_END})")
    print(f"  입력     : {DATA_DIR}")
    print(f"  출력     : {CACHE_DIR}")
    print(f"{'='*65}")

    if not os.path.isdir(DATA_DIR):
        print(f"[Error] 입력 폴더가 없습니다: {DATA_DIR}")
        exit(1)

    daily_bs = run_step1(DATA_DIR, CACHE_DIR)
    run_step2(daily_bs, CACHE_DIR)

    print(f"\n[완료] 소요 시간: {datetime.now() - start}")
    print(f"\n다음 단계: Rscript 01apin/02_r_apin.R")
