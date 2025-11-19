"""
CSV 파일 로더 모듈 (수정됨)
- context_text, start_date, end_date 컬럼 지원 추가
- 학사일정 데이터 포맷에 최적화
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
import re

import numpy as np
import pandas as pd
from langchain.schema import Document

# ===========================
# 내부 유틸
# ===========================
def _read_csv(csv_path: str | Path) -> pd.DataFrame:
    """CSV를 안전하게 로드 (인코딩/빈 줄 처리 등)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    # 일반적인 UTF-8, EUC-KR 케이스를 순차 시도
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"CSV 로드 실패: {path} (마지막 오류: {last_err})")

    # 전부 NaN인 행 제거
    df = df.dropna(how="all")
    return df


def _normalize_str(val: Any) -> str:
    """NaN / None을 안전하게 문자열로 변환."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip()


def _extract_date_info(date_str: str) -> Dict[str, Any]:
    """날짜 문자열 파싱 (백업용)"""
    if not isinstance(date_str, str):
        return {"start_date": None, "end_date": None}
    
    text = date_str.strip()
    if not text:
        return {"start_date": None, "end_date": None}

    # YYYY-MM-DD 패턴 찾기
    matches = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", text)
    if len(matches) >= 2:
        return {"start_date": matches[0], "end_date": matches[1]}
    elif len(matches) == 1:
        return {"start_date": matches[0], "end_date": matches[0]}
    
    return {"start_date": None, "end_date": None}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """주어진 후보 이름 리스트에서 실제 DataFrame 컬럼명을 1개 선택."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# ===========================
# 공개 API
# ===========================
def validate_csv_format(csv_path: str | Path) -> Dict[str, Any]:
    """
    CSV 형식이 유효한지 검사합니다.
    - context_text가 있거나, text/prep_text 컬럼이 있어야 함.
    """
    try:
        df = _read_csv(csv_path)
    except Exception as e:
        return {"valid": False, "message": f"CSV 읽기 실패: {e}", "file": str(csv_path)}

    cols = list(df.columns)

    # 1순위: 우리가 만든 context_text 확인
    if "context_text" in cols:
        return {"valid": True, "message": "OK (context_text 발견)", "file": str(csv_path), "columns": cols}

    # 2순위: 기존 방식 (text + title 조합)
    text_col = _pick_column(df, ["prep_text", "text", "내용", "description", "event"])
    if text_col is None:
        return {
            "valid": False, 
            "message": "텍스트 컬럼(context_text/prep_text/text/내용)이 없습니다.",
            "file": str(csv_path), "columns": cols
        }

    return {"valid": True, "message": "OK", "file": str(csv_path), "columns": cols}


def load_documents_from_csv(csv_path: str | Path) -> List[Document]:
    """
    CSV를 로드하여 Document 리스트로 변환합니다.
    - context_text 컬럼이 있으면 page_content로 바로 사용합니다.
    - start_date, end_date 컬럼이 있으면 메타데이터로 바로 사용합니다.
    """
    df = _read_csv(csv_path)
    docs: List[Document] = []

    # 컬럼 매핑 찾기
    col_context = _pick_column(df, ["context_text", "search_text"]) # 우리가 만든 검색용 텍스트
    col_text = _pick_column(df, ["prep_text", "text", "내용", "content"]) # 기존 텍스트
    col_title = _pick_column(df, ["title", "제목", "행사명", "date", "날짜"]) # 제목
    
    col_start = _pick_column(df, ["start_date", "시작일"])
    col_end = _pick_column(df, ["end_date", "종료일"])
    col_cat = _pick_column(df, ["category", "구분", "카테고리"])

    for _, row in df.iterrows():
        page_content = ""
        metadata = {"source": str(csv_path)}

        # 1. Page Content 결정
        if col_context and pd.notna(row[col_context]):
            # 전처리된 완성형 문장이 있으면 그대로 사용
            page_content = str(row[col_context])
        elif col_text and pd.notna(row[col_text]):
            # 없다면 기존 방식대로 조합
            title_val = str(row[col_title]) if col_title else ""
            text_val = str(row[col_text])
            page_content = f"{text_val}\n기간: {title_val}"
        else:
            continue # 내용이 없으면 스킵

        # 2. Metadata: 날짜
        start_date = None
        end_date = None

        # (1) 명시적 컬럼이 있는 경우
        if col_start and pd.notna(row[col_start]):
            start_date = str(row[col_start]).strip()
        if col_end and pd.notna(row[col_end]):
            end_date = str(row[col_end]).strip()

        # (2) 없는 경우 Title에서 추출 시도 (기존 방식 백업)
        if not start_date and col_title and pd.notna(row[col_title]):
             dates = _extract_date_info(str(row[col_title]))
             start_date = dates["start_date"]
             end_date = dates["end_date"]

        if start_date: metadata["start_date"] = start_date
        if end_date: metadata["end_date"] = end_date

        # 3. Metadata: 기타
        if col_title and pd.notna(row[col_title]):
            metadata["title"] = str(row[col_title]).strip()
        
        if col_cat and pd.notna(row[col_cat]):
            metadata["category"] = str(row[col_cat]).strip()

        # 문서 생성
        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs

#context_text 지원 추가:

# 이전 코드는 prep_text, text만 찾았습니다.

# 수정된 코드는 context_text가 있으면 가장 우선적으로 이것을 page_content(검색 대상)로 사용합니다.

# 명시적 날짜 컬럼(start_date, end_date) 지원:

# 이전 코드는 무조건 정규표현식으로 날짜를 다시 추출하려 했습니다.

# 수정된 코드는 CSV에 start_date가 있으면 그 값을 그대로 메타데이터에 넣습니다.

# 유효성 검사(validate_csv_format) 완화:

# context_text 컬럼만 있어도 "유효한 CSV"로 판단하도록 조건을 추가했습니다.