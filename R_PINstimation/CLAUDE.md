# CLAUDE.md — R_PINstimation 파이프라인 문서

이 파일은 `R_PINstimation/` 디렉토리의 코드 구조, 실행 흐름, 입출력 파일 형식을
Claude Code가 이해하고 작업하기 위한 레퍼런스 문서다.

---

## 1. 프로젝트 개요

한국·해외 주식시장 틱 데이터에서 **정보거래확률(PIN / APIN / VPIN)**을
R의 `PINstimation` 패키지로 계산하는 연구용 파이프라인.

| 모델 | 논문 | 파라미터 수 | 핵심 지표 |
|------|------|------------|----------|
| **PIN** | EKOP(1996) | 5개 (α, δ, μ, ε_b, ε_s) | PIN |
| **APIN** | Duarte & Young(2009) | 10개 (α, δ, θ, θ', μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s) | APIN, PSOS |
| **VPIN** | Easley et al.(2012) | BVC 기반 | VPIN |

각 모델은 **Python 전처리 → R 추정** 2단계로 구성된다.

---

## 2. 디렉토리 구조

```
R_PINstimation/
├── 00_sas_to_parquet.py     ← SAS7BDAT → Parquet 변환 (폴더 구조 그대로 유지)
├── 00pin/
│   ├── 01_preprocess.py     ← Step1: 틱→일별 B/S 집계, Step2: 캘린더 정렬
│   └── 02_r_pin.R           ← 60일 롤링 PIN 추정 (R, 병렬)
├── 01apin/
│   ├── 01_preprocess.py     ← 00pin과 공유 캐시 사용 (동일 로직)
│   └── 02_r_apin.R          ← 60일 롤링 APIN 추정 (R, 병렬)
├── 02vpin/
│   ├── 01_preprocess.py     ← 틱→1분봉 집계
│   └── 02_r_vpin.R          ← VPIN 추정 (R, 병렬)
└── Base_code/               ← 원형 코드 (검증용 16종목 샘플 버전)
    ├── 01_python_preprocessing.py
    └── 02_r_pinstimation.R
```

---

## 3. 입력 데이터 구조

### 3-1. 원본 틱 데이터 (BASE_DIR 하위)

SAS→parquet 변환 후의 폴더 구조를 PIN/APIN/VPIN 파이프라인이 그대로 읽는다.

```
E:\vpin_project_parquet\
├── KOR_2019\
│   ├── KOR_201901.parquet
│   ├── KOR_201902.parquet
│   └── ...  (월 1파일)
├── KOR_2020\
│   └── ...
└── KOR_2021\
    └── ...
```

**폴더명 규칙**: `{나라코드}_{연도}`  (SAS 원본 폴더명과 동일)

지원 나라코드: `KOR`, `US`, `JP`, `CA`, `FR`, `GR`, `HK`, `IT`, `UK`

**월별 parquet 컬럼** (PIN·APIN용):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| Symbol | Utf8 | 종목코드 (예: "A005930") |
| Date | Date | 거래일 |
| LR | Int | 매수=1, 매도=-1 (Lee-Ready 분류) |

**월별 parquet 컬럼** (VPIN용):

| 컬럼 | 타입 | 설명 |
|------|------|------|
| Symbol | Utf8 | 종목코드 |
| Date | Date | 거래일 |
| Time | Time | 거래시각 |
| Price | Float | 거래가격 |
| Volume | Float | 거래량 |

---

## 4. PIN 파이프라인 (`00pin/`)

### 4-1. Python 전처리 (`00pin/01_preprocess.py`)

**실행**: `python 00pin/01_preprocess.py`

**사용자 설정 구역** (파일 상단):
```python
BASE_DIR        = r"E:\vpin_project_parquet"   # 틱 parquet 루트
COUNTRY         = "KOR"                        # 나라코드 → KOR_YYYY 폴더 자동 스캔
FORCE_REPROCESS = False                        # True: 기존 캐시 무시하고 재생성
```

**Step 1 — 틱 데이터 → 일별 B/S 집계**

```
{COUNTRY}_YYYY 폴더 자동 스캔 → 월별 parquet N개
    ↓  Polars scan_parquet (lazy)
    ↓  LR 필터 (1 또는 -1만)
    ↓  group_by(Symbol, Date).agg(B=sum(LR==1), S=sum(LR==-1))
    ↓  sort(Symbol, Date)
→  all_daily_bs.parquet   (전 종목·전 기간 캐시)
```

- `FORCE_REPROCESS=False`이고 파일 존재 시 → **캐시 재사용**, Step1 스킵
- PIN·APIN이 **동일 캐시 파일을 공유**한다 (같은 `CACHE_DIR` 경로 사용)

**Step 2 — 영업일 캘린더 정렬**

```
all_daily_bs.parquet
    ↓  영업일 캘린더 = 전체 종목 기준 거래가 한 번이라도 있었던 날의 합집합
    ↓  청크 처리: CHUNK_SIZE=500 종목씩
       ① Polars cross join (500종목 × 전체 영업일)
       ② left join(실제 B/S) → 거래 없는 날 B=S=0 fill_null
       ③ PyArrow ParquetWriter.write_table() — 청크마다 즉시 파일에 기록
       ④ del aligned, table → 메모리 즉시 해제
→  full_daily_bs.parquet  (R 입력 파일)
```

메모리 피크: **청크당 ~25MB** (500종목 × 2500일 × 20bytes)

**Step 2 캘린더 정렬이 필요한 이유**: 특정 종목이 어떤 날 거래가 없으면 틱에 행 자체가
없다. 정렬 없이 60행 롤링 윈도우를 잡으면 "60행 = 여러 달"이 될 수 있다. 캘린더 정렬
후에는 60행이 항상 정확히 60 영업일을 의미한다.

**출력 파일**:

| 파일 | 경로 | 스키마 |
|------|------|--------|
| `all_daily_bs.parquet` | `R_output/{COUNTRY}/` | Symbol, Date, B(UInt32), S(UInt32) |
| `full_daily_bs.parquet` | `R_output/{COUNTRY}/` | Symbol, Date, B(Int32), S(Int32) |

---

### 4-2. R 추정 (`00pin/02_r_pin.R`)

**실행**: `Rscript 00pin/02_r_pin.R`

**사용자 설정 구역**:
```r
BASE_DIR    <- "E:/vpin_project_parquet"
COUNTRY <- "KOR"

WINDOW_SIZE    <- 60     # 롤링 윈도우 크기 (영업일)
MIN_VALID_DAYS <- 30     # 윈도우 내 실제 거래일(B+S>0) 최솟값

PIN_METHOD       <- "ML"   # "ML" 또는 "ECM"
PIN_INITIALSETS  <- "GE"   # "GE", "random" 등
NUM_INITIAL_SETS <- 20

NUM_WORKERS  <- parallel::detectCores(logical = TRUE)
CHECKPOINT_N <- 100   # N 종목마다 진행 로그 출력
```

**처리 흐름**:

```
[1] full_daily_bs.parquet 로드
[2] 체크포인트 확인
      checkpoints/sym_*.parquet 파일 목록 스캔
      → completed_syms = 이미 완료된 종목
      → remaining_syms = setdiff(all_symbols, completed_syms)
[3] 종목별 데이터 분할 (split_bs)
      remaining_syms 각각에 대해 args 리스트 구성
[4] 병렬 추정 — makeCluster(PSOCK) + parLapply
      워커당 1종목, 전체 기간 60일 롤링 pin() 반복
      윈도우마다: valid_days < MIN_VALID_DAYS 이면 스킵
                  pin() 수렴 실패 시 스킵
      완료 후: sym_{Symbol}.parquet 저장 (레이스컨디션 없음)
      CHECKPOINT_N 종목마다 진행률·ETA 로그 출력
[5] 체크포인트 병합
      모든 sym_*.parquet → rbind → 정렬 → 최종 저장
```

**워커 함수 `worker_process_symbol(args)`**:

- 독립 Rscript 프로세스(PSOCK)에서 실행 → 전역 변수 없음, args만 참조
- 내부에서 `library(PINstimation)`, `library(arrow)` 직접 로드
- 완료 즉시 `checkpoints/sym_{Symbol}.parquet` 저장
- 반환값: `list(symbol=..., n_rows=...)`

**출력 파일**:

| 파일 | 경로 |
|------|------|
| `checkpoints/sym_{Symbol}.parquet` | `R_output/{COUNTRY}/pin/checkpoints/` |
| `pin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet` | `R_output/{COUNTRY}/pin/` |
| `pin_{COUNTRY}_{YYYYMMDD_HHMM}.csv` | `R_output/{COUNTRY}/pin/` |

**결과 스키마** (PIN):

| 컬럼 | 설명 |
|------|------|
| Symbol | 종목코드 |
| Date | 윈도우 마지막 날짜 (= 추정 기준일) |
| valid_days | 윈도우 내 실제 거래일 수 (30~60) |
| a | alpha — 정보 이벤트 발생 확률 |
| d | delta — 호재 조건부 확률 |
| u | mu — 정보 거래 도착률 |
| eb | eps_b — 비정보 매수 도착률 |
| es | eps_s — 비정보 매도 도착률 |
| PIN | 정보거래확률 |

---

## 5. APIN 파이프라인 (`01apin/`)

### 5-1. Python 전처리 (`01apin/01_preprocess.py`)

**실행**: `python 01apin/01_preprocess.py`

PIN 전처리와 **완전히 동일한 로직**. `CACHE_DIR`이 같은 경로이므로 PIN 전처리를
먼저 실행했다면 `all_daily_bs.parquet`·`full_daily_bs.parquet` 캐시를 자동 재사용한다.

사용자 설정:
```python
BASE_DIR    = r"E:\vpin_project_parquet"
COUNTRY = "KOR"
FORCE_REPROCESS = False
```

---

### 5-2. R 추정 (`01apin/02_r_apin.R`)

**실행**: `Rscript 01apin/02_r_apin.R`

PIN과 구조 동일. `pin()` 대신 `adjpin()` 사용.

**추가 설정**:
```r
ADJPIN_METHOD      <- "ECM"   # ECM: ML보다 빠르고 수렴 안정적
ADJPIN_INITIALSETS <- "GE"    # GE: Ghachem & Ersan 표준 초기값
NUM_INITIAL_SETS   <- 20
```

**결과 스키마** (APIN — 15컬럼):

| 컬럼 | PINstimation 파라미터명 | 설명 |
|------|------------------------|------|
| Symbol | — | 종목코드 |
| Date | — | 추정 기준일 |
| valid_days | — | 윈도우 내 실제 거래일 수 |
| a | alpha | 정보 이벤트 발생 확률 |
| d | delta | 호재 조건부 확률 |
| t1 | theta | 무정보일 SPOS 충격 확률 |
| t2 | thetap | 정보일 SPOS 충격 확률 |
| ub | mu.b | 호재 정보 매수 추가 도착률 |
| us | mu.s | 악재 정보 매도 추가 도착률 |
| eb | eps.b | 비정보 매수 기본 도착률 |
| es | eps.s | 비정보 매도 기본 도착률 |
| pb | d.b | SPOS 충격 추가 매수량 Δ_b |
| ps | d.s | SPOS 충격 추가 매도량 Δ_s |
| APIN | — | Adjusted PIN |
| PSOS | — | Probability of Symmetric Order-flow Shock |

**출력 파일**:

| 파일 | 경로 |
|------|------|
| `checkpoints/sym_{Symbol}.parquet` | `R_output/{COUNTRY}/apin/checkpoints/` |
| `apin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet` | `R_output/{COUNTRY}/apin/` |
| `apin_{COUNTRY}_{YYYYMMDD_HHMM}.csv` | `R_output/{COUNTRY}/apin/` |

---

## 6. VPIN 파이프라인 (`02vpin/`)

PIN·APIN과 달리 일별 B/S가 아니라 **원시 가격·거래량**이 필요하므로 별도 전처리를 수행한다.

### 6-1. Python 전처리 (`02vpin/01_preprocess.py`)

**실행**: `python 02vpin/01_preprocess.py`

**사용자 설정**:
```python
BASE_DIR        = r"E:\vpin_project_parquet"
COUNTRY         = "KOR"
FORCE_REPROCESS = False
```

**처리 흐름**:

```
{COUNTRY}_YYYY 폴더 자동 스캔 → 월별 parquet N개
    ↓  Polars scan_parquet (lazy)
    ↓  컬럼 선택: Symbol, Date, Time, Price, Volume
    ↓  Volume > 0 필터
    ↓  Datetime = Date.combine(Time) 생성
    ↓  group_by_dynamic(Datetime, every="1m", group_by=Symbol)
       .agg(Price=last(), Volume=sum())  ← 1분봉 집계
    ↓  sort(Symbol, Datetime)
→  all_1m_bars.parquet  (1분봉 집계 완료)
```

**출력 파일**:

| 파일 | 경로 | 스키마 |
|------|------|--------|
| `all_1m_bars.parquet` | `R_output/{COUNTRY}/vpin/` | Symbol, Datetime, Price(Float64), Volume(Float64) |

---

### 6-2. R 추정 (`02vpin/02_r_vpin.R`)

**실행**: `Rscript 02vpin/02_r_vpin.R`

**사용자 설정**:
```r
BASE_DIR    <- "E:/vpin_project_parquet"
COUNTRY <- "KOR"

ROLLING_WINDOW  <- 50    # VPIN 롤링 버킷 수
BUCKETS_PER_DAY <- 50    # 버킷 크기 V = ADV / BUCKETS_PER_DAY
SESSION_LENGTH  <- 6.5   # 한국: 09:00~15:30 = 6.5시간
TIMEBARSIZE     <- 1     # 1분봉 입력
```

**처리 흐름**:

```
[1] all_1m_bars.parquet 로드
      Datetime → as.POSIXct(tz="Asia/Seoul") 변환
[2] 체크포인트 확인 (PIN·APIN과 동일 메커니즘)
[3] 종목별 데이터 분할 + args 구성
[4] 병렬 추정 — makeCluster(PSOCK) + parLapply
      워커: vpin(data=..., sessionlength=6.5, buckets=50,
                 window=50, timebarsize=1, verbose=FALSE)
      결과 S4 객체에서 @data (버킷 단위) 또는 @vpin (벡터) 추출
      sym_{Symbol}.parquet 저장
[5] 체크포인트 병합 → 최종 저장
```

**결과 스키마** (VPIN):

| 컬럼 | 설명 |
|------|------|
| Symbol | 종목코드 |
| Datetime | 버킷 종료 시각 |
| BucketNo | 버킷 번호 (1부터) |
| VPIN | Volume-Synchronized PIN |

**출력 파일**:

| 파일 | 경로 |
|------|------|
| `checkpoints/sym_{Symbol}.parquet` | `R_output/{COUNTRY}/vpin/checkpoints/` |
| `vpin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet` | `R_output/{COUNTRY}/vpin/` |
| `vpin_{COUNTRY}_{YYYYMMDD_HHMM}.csv` | `R_output/{COUNTRY}/vpin/` |

---

## 7. 전체 폴더 구조

```
E:\vpin_project_parquet\          ← BASE_DIR
│
├── KOR_2019\                     ← SAS→parquet 출력 (연도별 폴더)
│   ├── KOR_201901.parquet
│   ├── KOR_201902.parquet
│   └── ...
├── KOR_2020\
│   └── ...
│
└── R_output\
    └── KOR\                      ← COUNTRY별 출력 루트
        │
        ├── all_daily_bs.parquet          ← PIN·APIN 공유 캐시 (Step1 출력)
        ├── full_daily_bs.parquet         ← 캘린더 정렬 완료 (Step2 출력, R 입력)
        │
        ├── pin\
        │   ├── checkpoints\
        │   │   ├── sym_A005930.parquet   ← 완료 종목별 체크포인트
        │   │   └── ...
        │   ├── pin_KOR_20260310_1430.parquet   ← 최종 결과 (모델·나라·완료일시)
        │   └── pin_KOR_20260310_1430.csv
        │
        ├── apin\
        │   ├── checkpoints\
        │   │   └── sym_*.parquet
        │   ├── apin_KOR_20260310_1430.parquet
        │   └── apin_KOR_20260310_1430.csv
        │
        └── vpin\
            ├── all_1m_bars.parquet        ← VPIN 전처리 출력 (R 입력)
            ├── checkpoints\
            │   └── sym_*.parquet
            ├── vpin_KOR_20260310_1430.parquet
            └── vpin_KOR_20260310_1430.csv
```

---

## 8. 체크포인트·재개 메커니즘

### 체크포인트 저장

R 워커가 1종목 전체 기간 계산 완료 후 즉시 개별 파일 저장:
```
checkpoints/sym_{Symbol}.parquet
```

각 워커가 **각자 독립 파일**에 쓰므로 레이스컨디션이 발생하지 않는다.

### 중단 재개

R 스크립트 재실행 시 자동으로:
```r
ckpt_files     <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$")
completed_syms <- gsub("^sym_|\\.parquet$", "", ckpt_files)
remaining_syms <- setdiff(all_symbols, completed_syms)
```
→ 완료된 종목은 건너뛰고, 미완료 종목만 계산

### 최종 병합

체크포인트 디렉토리의 모든 `sym_*.parquet`를 `rbind` 후 정렬하여 최종 파일 저장.
**체크포인트 파일은 삭제하지 않는다** (다음 재실행에서 재사용 가능).

---

## 9. 병렬 처리 구조

```
메인 프로세스 (R)
    │
    ├─ makeCluster(n_workers, type="PSOCK")
    │   ├─ Worker 1 (독립 Rscript) ─ sym_A005930 처리
    │   ├─ Worker 2 (독립 Rscript) ─ sym_A000660 처리
    │   ├─ ...
    │   └─ Worker N (독립 Rscript) ─ sym_XXXXXX 처리
    │
    └─ parLapply(cl, symbol_args[batch], worker_process_symbol)
         ※ CHECKPOINT_N 종목 단위 배치로 나눠 진행 로그 출력
```

- **PSOCK 방식**: Windows·Linux·macOS 모두 호환
- **메모리 독립**: 각 워커는 완전히 독립된 메모리 공간, 공유 변수 없음
- **1코어 1종목**: 한 워커가 한 종목의 전체 기간(모든 롤링 윈도우)을 완주
- `on.exit(stopCluster(cl))` → 정상 완료·오류·중단 어떤 경우에도 워커 프로세스 정리

---

## 10. 실행 순서 요약

### PIN 계산
```bash
python 00pin/01_preprocess.py
Rscript 00pin/02_r_pin.R
```

### APIN 계산 (PIN 전처리 완료 후 캐시 재사용 가능)
```bash
python 01apin/01_preprocess.py   # all_daily_bs.parquet 이미 있으면 스킵
Rscript 01apin/02_r_apin.R
```

### VPIN 계산
```bash
python 02vpin/01_preprocess.py
Rscript 02vpin/02_r_vpin.R
```

---

## 11. 파일명 버전 관리

최종 결과 파일명 규칙:
```
{모델}_{COUNTRY}_{완료일시}.{확장자}

예) pin_KOR_20260310_1430.parquet
    apin_US_20260315_0900.csv
    vpin_JP_20260320_1200.parquet
```

- `{모델}` → `pin` / `apin` / `vpin` — 파일명만 봐도 어떤 모델인지 즉시 식별
- `{COUNTRY}` → 나라코드 — 어느 나라 데이터인지 식별
- `{완료일시}` → `YYYYMMDD_HHMM` 형식 → 동일 나라 재실행 시 이전 결과를 덮어쓰지 않음

---

## 12. 의존 패키지

### Python
```
polars, pyarrow, glob, math, re, os, warnings
```

### R
```r
install.packages(c("PINstimation", "arrow", "parallel"))
# parallel은 R 기본 내장 패키지
```

---

## 13. 공통 설계 원칙

1. **사용자 설정 구역 분리**: 각 파일 상단 `★ 사용자 설정 구역`만 수정하면 실행 가능
2. **캐시 우선**: `FORCE_REPROCESS=False`이면 기존 파일 재사용 → 불필요한 재계산 방지
3. **메모리 효율**: Python Step2는 500종목 청크 + PyArrow 증분 쓰기 → 피크 메모리 최소화
4. **레이스컨디션 없음**: 워커별 독립 파일 쓰기 → 잠금 없이 안전한 병렬 처리
5. **중단 내성**: 종목 단위 체크포인트 → 언제든 중단해도 완료 종목부터 재개
6. **버전 관리**: 완료일시 포함 파일명 → 여러 실행 결과 공존 가능

---

## 14. Base_code/ (원형 코드)

`Base_code/`는 16개 샘플 종목만 계산하던 **초기 검증용** 버전이다.

- `01_python_preprocessing.py`: `SAMPLE_SYMBOLS` 리스트 16개 종목 필터, 연도 필터
- `02_r_pinstimation.R`: APIN만 계산, 체크포인트 없음, 결과를 `r_apin_results.parquet`로 단순 저장

현재 운영 버전(`00pin/`, `01apin/`, `03vpin/`)과 구조가 다르므로 참조용으로만 사용.
