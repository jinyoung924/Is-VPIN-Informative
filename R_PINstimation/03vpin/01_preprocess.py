"""
=============================================================================
[VPIN 파이프라인 - Step 1] Python 전처리
틱 데이터 (parquet 파일들) → 1분봉 집계 → all_1m_bars.parquet
=============================================================================

PIN/APIN과 달리 원시 가격·거래량이 필요하므로 별도 전처리를 수행한다.

지원 입력 폴더 형식: {나라코드}_{시작연도월}_{종료연도월}
  예) KOR_201910_202107  /  US_201901_202012  /  JP_202001_202312

지원 나라코드: KOR  US  JP  CA  FR  GR  HK  IT  UK

실행 순서:
  1) python 03vpin/01_preprocess.py
  2) Rscript 03vpin/02_r_vpin.R

=============================================================================
"""

import os
import re
import sys
import glob
import polars as pl
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

# True → 기존 all_1m_bars.parquet 가 있어도 강제 재생성
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

DATA_DIR   = os.path.join(BASE_DIR, DATA_FOLDER)
OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", DATA_FOLDER, "vpin")


# =============================================================================
# Step 1: 틱 데이터 → 1분봉
# =============================================================================

def process_file_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """틱 parquet 파일 1개를 읽어 1분봉으로 집계한다."""
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "Time", "Price", "Volume"])
        .filter(pl.col("Volume") > 0)
        .with_columns(pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime"))
        .drop(["Date", "Time"])
        .sort(["Symbol", "Datetime"])
        .collect()
    )

    if df.is_empty():
        print("0 봉")
        return pl.DataFrame(schema={
            "Symbol": pl.String, "Datetime": pl.Datetime,
            "Price": pl.Float64, "Volume": pl.Float64,
        })

    bars = (
        df
        .group_by_dynamic("Datetime", every="1m", group_by="Symbol", closed="left")
        .agg([
            pl.col("Price").last().alias("Price"),
            pl.col("Volume").sum().alias("Volume"),
        ])
        .select(["Symbol", "Datetime", "Price", "Volume"])
        .sort(["Symbol", "Datetime"])
    )
    print(f"{bars.height:,} 봉")
    return bars


def get_parquet_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    print(f"\n[파일 탐색] {data_dir}")
    print(f"  parquet {len(files)}개 발견")
    return files


def run_preprocessing(data_dir: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_1m_bars.parquet")

    if not FORCE_REPROCESS and os.path.exists(output_path):
        n = pl.scan_parquet(output_path).select(pl.len()).collect()[0, 0]
        print(f"\n[Step 1 스킵] 기존 파일 재사용: {output_path}  ({n:,} 행)")
        return output_path

    parquet_files = get_parquet_files(data_dir)
    if not parquet_files:
        raise RuntimeError(f"[Error] parquet 파일을 찾을 수 없습니다: {data_dir}")

    print(f"\n{'='*65}")
    print(f"[Step 1] 틱 데이터 → 1분봉 집계  [{DATA_FOLDER}]")
    print(f"  출력 경로: {output_path}")
    print(f"{'='*65}\n")

    all_bars: List[pl.DataFrame] = []
    for path in parquet_files:
        bars = process_file_to_1m_bars(path)
        if not bars.is_empty():
            all_bars.append(bars)

    if not all_bars:
        raise RuntimeError("[Error] 유효한 봉 데이터가 없습니다.")

    print(f"\n  파일 {len(all_bars)}개 병합 중...")
    full_bars = (
        pl.concat(all_bars, how="vertical")
        .sort(["Symbol", "Datetime"])
    )
    full_bars.write_parquet(output_path, compression="zstd")

    print(f"\n[Step 1 완료] {output_path}")
    print(f"  총 봉 수 : {full_bars.height:,}")
    print(f"  종목 수  : {full_bars['Symbol'].n_unique():,}")
    print(f"  시간 범위: {full_bars['Datetime'].min()} ~ {full_bars['Datetime'].max()}")
    return output_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN 전처리] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  데이터   : {DATA_FOLDER}  (나라: {COUNTRY}, 기간: {PERIOD_START}~{PERIOD_END})")
    print(f"  입력     : {DATA_DIR}")
    print(f"  출력     : {OUTPUT_DIR}")
    print(f"{'='*65}")

    if not os.path.isdir(DATA_DIR):
        print(f"[Error] 입력 폴더가 없습니다: {DATA_DIR}")
        exit(1)

    run_preprocessing(DATA_DIR, OUTPUT_DIR)

    print(f"\n[완료] 소요 시간: {datetime.now() - start}")
    print(f"\n다음 단계: Rscript 03vpin/02_r_vpin.R")
