import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import re
import time
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.figure_factory as ff

# (Tuỳ chọn) AI insight
try:
    import google.generativeai as genai
except Exception:
    genai = None

# =========================
# Config
# =========================
WB_BASE = "https://api.worldbank.org/v2"
WB_VECTOR_SEARCH_URL = "https://data360api.worldbank.org/data360/searchv2"
HEADERS = {"User-Agent": "Streamlit-WB-Client/1.0 (contact: you@example.com)",
           "Accept": "application/json"}
POST_HEADERS = {**HEADERS, "Content-Type": "application/json"}
REQ_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF     = 1.6
DEFAULT_DATE_RANGE = (2000, 2024)

# =========================
# Helpers (retry)
# =========================

def _sleep(attempt: int, base: float = BACKOFF) -> float:
    return min(base ** attempt, 12.0)


def http_get_json(url: str, params: Dict[str, Any]) -> Any:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(_sleep(attempt))
    raise RuntimeError(f"GET {url} failed after retries: {last_err}")


def http_post_json(url: str, payload: Dict[str, Any]) -> Any:
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=payload, headers=POST_HEADERS, timeout=REQ_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(_sleep(attempt))
    raise RuntimeError(f"POST {url} failed after retries: {last_err}")

# =========================
# Indicator utilities
# =========================
_VALID_WB_ID = re.compile(r"^[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)+$")


def is_valid_wb_id(candidate: str) -> bool:
    if not isinstance(candidate, str):
        return False
    c = candidate.strip()
    return bool(_VALID_WB_ID.match(c))


def normalize_indicator_id(raw_id: str) -> Optional[str]:
    if not isinstance(raw_id, str):
        return None
    candidate = raw_id.strip().upper()
    if not candidate:
        return None
    if candidate.startswith("WB_"):
        parts = candidate.split("_")
        if len(parts) >= 3:
            candidate = "_".join(parts[2:])
        else:
            candidate = parts[-1]
    candidate = candidate.replace("_", ".")
    candidate = re.sub(r"\.+", ".", candidate)
    candidate = candidate.strip(".")
    if not candidate:
        return None
    return candidate if is_valid_wb_id(candidate) else None


@st.cache_data(show_spinner=False, ttl=24*3600)
def wb_search_indicators(keyword: str, top: int = 50) -> pd.DataFrame:
    search_term = (keyword or "").strip()
    if not search_term:
        return pd.DataFrame(columns=["id", "raw_id", "name", "source", "score"])
    limit = max(1, min(int(top or 1), 500))
    payload = {
        "count": True,
        "select": "series_description/idno, series_description/name, series_description/database_id",
        "search": search_term,
        "top": limit
    }
    js = http_post_json(WB_VECTOR_SEARCH_URL, payload)
    values = (js or {}).get("value", [])
    results: List[Dict[str, str]] = []
    for item in values or []:
        series_desc = (item or {}).get("series_description") or {}
        raw_id = (series_desc.get("idno") or "").strip()
        normalized = normalize_indicator_id(raw_id)
        if not normalized:
            continue
        score = item.get("@search.score", 0.0)
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            score_val = 0.0
        results.append({
            "id": normalized,
            "raw_id": raw_id,
            "name": (series_desc.get("name") or "").strip() or normalized,
            "source": (series_desc.get("database_id") or "").strip() or "N/A",
            "score": score_val,
        })
    df = pd.DataFrame(results)
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["id"])
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

# =========================
# Fetch series
# =========================
@st.cache_data(show_spinner=False, ttl=1200)
def wb_fetch_series(country_code: str, indicator_id: str, year_from: int, year_to: int) -> pd.DataFrame:
    js = http_get_json(
        f"{WB_BASE}/country/{country_code}/indicator/{indicator_id}",
        {"format": "json", "per_page": 20000, "date": f"{int(year_from)}:{int(year_to)}"}
    )

    if not isinstance(js, list) or len(js) < 2:
        return pd.DataFrame(columns=["Year", "Country", "IndicatorID", "Value"])
    if isinstance(js[0], dict) and js[0].get("message"):
        return pd.DataFrame(columns=["Year", "Country", "IndicatorID", "Value"])

    _, data = js
    rows = []
    for d in (data or []):
        year_raw = str(d.get("date", ""))
        year = int(year_raw) if year_raw.isdigit() else None
        rows.append({
            "Year": year,
            "Country": (d.get("country") or {}).get("value", country_code),
            "IndicatorID": (d.get("indicator") or {}).get("id", indicator_id),
            "Value": d.get("value", None)
        })
    out = pd.DataFrame(rows).dropna(subset=["Year"]) if rows else pd.DataFrame(columns=["Year","Country","IndicatorID","Value"])
    return out.sort_values(["Country","IndicatorID","Year"]) if not out.empty else out


def pivot_wide(df_long: pd.DataFrame, use_friendly_name: bool, id_to_name: Dict[str, str]) -> pd.DataFrame:
    if df_long is None or df_long.empty:
        return pd.DataFrame()
    key_col = "IndicatorName" if use_friendly_name else "IndicatorID"
    df = df_long.copy()
    if use_friendly_name:
        df["IndicatorName"] = df["IndicatorID"].map(id_to_name).fillna(df["IndicatorID"])
    wide = df.pivot_table(index=["Year","Country"], columns=key_col, values="Value", aggfunc="first")
    wide = wide.reset_index().sort_values(["Country","Year"])
    wide = wide.rename(columns={"Year": "Năm"})
    return wide

# =========================
# Data utilities
# =========================

def handle_na(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if method == "Giữ nguyên (N/A)":
        return df
    if method == "Điền 0":
        return df.fillna(0)
    if method == "Forward-fill theo Country + cột dữ liệu":
        cols = [c for c in df.columns if c not in ("Năm", "Country")]
        return (df.sort_values(["Country","Năm"]) \
                  .groupby("Country")[cols] \
                  .ffill() \
                  .reindex(df.index) \
                  .pipe(lambda d: df.assign(**{c: d[c] for c in cols})))
    if method == "Backward-fill theo Country + cột dữ liệu":
        cols = [c for c in df.columns if c not in ("Năm", "Country")]
        return (df.sort_values(["Country","Năm"]) \
                  .groupby("Country")[cols] \
                  .bfill() \
                  .reindex(df.index) \
                  .pipe(lambda d: df.assign(**{c: d[c] for c in cols})))
    return df

# =========================
# UI
# =========================

st.set_page_config(page_title="World Bank WDI — Sửa python7", layout="wide")
st.title("Công cụ tổng hợp và phân tích dữ liệu vĩ mô kết hợp AI")
st.caption("Tìm indicator (WDI, lọc ID hợp lệ) → Lấy dữ liệu qua API v2 → Bảng rộng: Năm, Country, chỉ số…")

# ===== Sidebar: Tool tìm indicator, chọn năm, Xử lý N/A, Quốc gia =====
with st.sidebar:
    st.header("🔧 Công cụ")
    # Quốc gia
    country_raw = st.text_input("Country codes (ISO2/3, ',' tách)", value="VN")

    # Tìm indicator
    st.subheader("Tìm chỉ số")
    kw = st.text_input("Từ khoá", value="GDP")
    top_n = st.number_input("Top", 1, 500, 50, 1)
    do_search = st.button("🔍 Tìm indicator")

    if do_search:
        if not kw.strip():
            st.warning("Nhập từ khoá trước khi tìm.")
        else:
            with st.spinner("Đang tìm indicator…"):
                df_ind = wb_search_indicators(kw.strip(), top=int(top_n))
                st.session_state["ind_search_df"] = df_ind
                st.session_state["selected_indicator_ids"] = []
                st.session_state["select_all_indicator_results"] = False

    # Khoảng năm + xử lý NA
    y_from, y_to = st.slider("Khoảng năm", 1995, 2025, DEFAULT_DATE_RANGE)
    na_method = st.selectbox(
        "Xử lý N/A",
        [
            "Giữ nguyên (N/A)",
            "Điền 0",
            "Forward-fill theo Country + cột dữ liệu",
            "Backward-fill theo Country + cột dữ liệu",
        ],
        index=0,
    )

    # Nút tải dữ liệu
    load_clicked = st.button("📥 Tải dữ liệu")

# ===== Main area: Tabs riêng biệt =====
TAB_TITLES = ["📊 Dữ liệu", "📈 Biểu đồ", "🧮 Thống kê", "📥 Xuất dữ liệu", "🤖 AI"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_TITLES)

# Tải kết quả tìm kiếm để chọn indicator
ind_df = st.session_state.get("ind_search_df", pd.DataFrame())
id_to_name = {row["id"]: row["name"] for _, row in (ind_df if not ind_df.empty else pd.DataFrame()).iterrows()}
selected_indicator_ids = st.session_state.get("selected_indicator_ids", [])
select_all_requested = st.session_state.get("select_all_indicator_results", False)
if select_all_requested and not ind_df.empty:
    all_ids = ind_df["id"].dropna().tolist()
    if set(selected_indicator_ids) != set(all_ids):
        selected_indicator_ids = all_ids
        st.session_state["selected_indicator_ids"] = selected_indicator_ids

with tab1:
    st.subheader("Chọn chỉ số từ kết quả tìm kiếm")
    if ind_df.empty:
        st.info("Hãy dùng thanh bên trái để *Tìm indicator*.")
    else:
        selected_set = set(selected_indicator_ids)
        if "score" in ind_df.columns:
            sortable_df = ind_df.sort_values("score", ascending=False, na_position="last").copy()
        else:
            sortable_df = ind_df.copy()
        display_df = (
            sortable_df
            .set_index("id")[["name", "source"]]
            .rename(columns={"name": "Tên indicator", "source": "Source"})
            .copy()
        )
        checkbox_col = "Chọn"
        display_df.insert(0, checkbox_col, display_df.index.isin(selected_set))
        edited_df = st.data_editor(
            display_df,
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                checkbox_col: st.column_config.CheckboxColumn(label="Chọn", default=False),
                "Tên indicator": st.column_config.TextColumn("Tên indicator"),
                "Source": st.column_config.TextColumn("Source"),
            },
        )
        selected_indicator_ids = edited_df.index[edited_df[checkbox_col]].tolist()
        st.session_state["selected_indicator_ids"] = selected_indicator_ids
        st.checkbox("Chọn tất cả các chỉ tiêu tìm thấy", key="select_all_indicator_results")

    if load_clicked:
        if not selected_indicator_ids:
            st.warning("Chọn ít nhất một chỉ số.")
            st.stop()
        if country_raw.strip().upper() == "ALL":
            country_list = ["all"]
        else:
            country_list = [c.strip() for c in country_raw.split(",") if c.strip()]
        chosen_ids = [cid for cid in selected_indicator_ids if cid and is_valid_wb_id(cid)]
        if not chosen_ids:
            st.error("Không có ID hợp lệ sau khi lọc.")
            st.stop()
        all_long: List[pd.DataFrame] = []
        with st.spinner(f"Đang tải {len(chosen_ids)} chỉ số…"):
            for country in country_list:
                for ind_id in chosen_ids:
                    df_fetch = wb_fetch_series(country, ind_id, int(y_from), int(y_to))
                    if df_fetch is not None and not df_fetch.empty:
                        all_long.append(df_fetch)
                    time.sleep(0.25)
        if not all_long:
            st.info("Không có dữ liệu phù hợp.")
            st.stop()
        df_long = pd.concat(all_long, ignore_index=True)
        df_wide = pivot_wide(df_long, use_friendly_name=True, id_to_name=id_to_name)
        df_wide = handle_na(df_wide, na_method)
        st.session_state["wb_df_wide"] = df_wide
        st.success("✅ Đã tải và hợp nhất dữ liệu.")

    df_show = st.session_state.get("wb_df_wide", pd.DataFrame())
    if not df_show.empty:
        st.dataframe(df_show.set_index(["Country","Năm"]), use_container_width=True)


def _get_df_wide() -> pd.DataFrame:
    return st.session_state.get("wb_df_wide", pd.DataFrame())

with tab2:
    st.subheader("Biểu đồ xu hướng")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu. Vào tab **Dữ liệu** để tải.")
    else:
        value_cols = [c for c in df.columns if c not in ("Năm", "Country")]
        df_long_plot = df.melt(id_vars=["Năm","Country"], value_vars=value_cols,
                               var_name="Indicator", value_name="Value")
        choose = st.multiselect("Chọn chỉ số để vẽ", options=sorted(value_cols), default=value_cols[:min(4, len(value_cols))])
        if choose:
            df_plot = df_long_plot[df_long_plot["Indicator"].isin(choose)].copy()
            fig = px.line(df_plot.sort_values(["Country","Indicator","Năm"]),
                          x="Năm", y="Value", color="Indicator", line_group="Country",
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)

            if len(choose) > 1:
                df_sel = df[choose].apply(pd.to_numeric, errors="coerce")
                df_sel = df_sel.dropna(axis=1, how="all")
                if df_sel.shape[1] >= 2:
                    corr = df_sel.corr().fillna(0)
                    hm = ff.create_annotated_heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        annotation_text=corr.round(2).values,
                        showscale=True,
                    )
                    st.plotly_chart(hm, use_container_width=True)

with tab3:
    st.subheader("Thống kê mô tả")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        cols = [c for c in df.columns if c not in ("Năm", "Country")]
        if not cols:
            st.info("Không có cột số để thống kê.")
        else:
            stats = df[cols].apply(pd.to_numeric, errors="coerce").describe().T
            stats["CV"] = (stats["std"]/stats["mean"]).abs()
            st.dataframe(
                stats[["mean","std","min","50%","max","CV"]]
                .rename(columns={"mean":"Mean","std":"Std","50%":"Median"}),
                use_container_width=True
            )

with tab4:
    st.subheader("Tải CSV")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        st.download_button(
            "💾 Tải CSV",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="worldbank_wdi_wide.csv",
            mime="text/csv",
        )

with tab5:
    st.subheader("AI Insight")
    df = _get_df_wide()
    if df.empty:
        st.info("Chưa có dữ liệu — hãy tải ở tab **Dữ liệu**.")
    else:
        target_audience = st.selectbox("Đối tượng tư vấn", ["Ngân hàng Agribank"])
        if genai is None or not (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY")):
            st.info("Chưa cấu hình GEMINI_API_KEY nên bỏ qua AI insight.")
        else:
            if st.button("🚀 Sinh AI phân tích"):
                try:
                    api_key = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY"))
                    genai.configure(api_key=api_key)
                    model_name = "gemini-2.5-pro"
                    model = genai.GenerativeModel(model_name)
                    data_csv = df.to_csv(index=False)
                    prompt = f"""
Bạn là chuyên gia kinh tế vĩ mô. Dữ liệu World Bank (định dạng wide):

{data_csv}

Hãy tóm tắt xu hướng chính, điểm bất thường, và gợi ý 2–3 khuyến nghị hành động cho đối tượng : {target_audience}.
Trình bày ngắn gọn theo gạch đầu dòng
**1. Bối cảnh & Dữ liệu chính:**
                Tóm tắt ngắn gọn bối cảnh kinh tế.Nêu bật các chỉ số chính và mức trung bình của chúng.

                **2. Xu hướng nổi bật & Biến động:**
                Phân tích các xu hướng tăng/giảm rõ rệt nhất (ví dụ: GDP, Xuất khẩu). Chỉ ra những năm có biến động mạnh nhất (ví dụ: Lạm phát) và giải thích ngắn gọn nguyên nhân nếu có thể.

                **3. Tương quan đáng chú ý:**
                Chỉ ra các mối tương quan thú vị (ví dụ: Tăng trưởng GDP và FDI, Lạm phát và Lãi suất...). Diễn giải ý nghĩa của các mối tương quan này.

                **4. Kiến nghị cho đối tượng: {target_audience}**
                Cung cấp 3-4 kiến nghị chiến lược, cụ thể, hữu ích và trực tiếp liên quan đến đối tượng 
                **5. Hành động thực thi (kèm KPI/Điều kiện kích hoạt):**
                Từ các kiến nghị ở mục 4, đề xuất 1-2 hành động cụ thể mà **{target_audience}** có thể thực hiện ngay. Gắn chúng với một KPI (Chỉ số đo lường hiệu quả) hoặc một "Điều kiện kích hoạt" (Trigger)..
"""
                    with st.spinner("AI đang phân tích…"):
                        resp = model.generate_content(prompt)
                        st.markdown(resp.text or "_Không có phản hồi_")
                except Exception as e:
                    st.warning(f"AI lỗi: {e}")
