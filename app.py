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
HEADERS = {"User-Agent": "Streamlit-WB-Client/1.0 (contact: you@example.com)",
           "Accept": "application/json"}
REQ_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF     = 1.6
DEFAULT_DATE_RANGE = (2000, 2024)

COUNTRY_OPTIONS = [
    ("Toàn cầu (ALL)", "all"),
    ("Việt Nam (VN)", "VN"),
    ("Hoa Kỳ (US)", "US"),
    ("Nhật Bản (JP)", "JP"),
    ("Singapore (SG)", "SG"),
    ("Thái Lan (TH)", "TH"),
    ("Hàn Quốc (KR)", "KR"),
    ("Trung Quốc (CN)", "CN"),
    ("Khu vực Euro (EUU)", "EUU"),
    ("Liên minh Châu Âu (EU)", "EU"),
    ("Anh (GB)", "GB"),
    ("Đức (DE)", "DE"),
    ("Pháp (FR)", "FR"),
    ("Canada (CA)", "CA"),
    ("Úc (AU)", "AU"),
    ("Ấn Độ (IN)", "IN"),
    ("Indonesia (ID)", "ID"),
    ("Malaysia (MY)", "MY"),
    ("Philippines (PH)", "PH"),
    ("Brazil (BR)", "BR"),
]
COUNTRY_LABEL_TO_CODE = dict(COUNTRY_OPTIONS)

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

# =========================
# Indicator utilities
# =========================
_VALID_WB_ID = re.compile(r"^[A-Z][A-Z0-9_.-]*$")


def is_valid_wb_id(candidate: str) -> bool:
    if not isinstance(candidate, str):
        return False
    c = candidate.strip()
    return bool(_VALID_WB_ID.match(c))


def normalize_indicator_id(raw_id: str, database_id: Optional[str] = None) -> Optional[str]:
    """Loại bỏ tiền tố dataset (vd: WB_WDI_) và chuyển '_' thành '.'."""
    if not isinstance(raw_id, str):
        return None
    code = raw_id.strip().upper()
    if not code:
        return None
    db_prefix = (database_id or "").strip().upper()
    if db_prefix and code.startswith(f"{db_prefix}_"):
        code = code[len(db_prefix) + 1 :]
    elif code.startswith("WB_"):
        parts = code.split("_", 2)
        if len(parts) == 3:
            code = parts[2]
        elif len(parts) > 3:
            code = "_".join(parts[2:])
        else:
            code = code.replace("WB_", "", 1)
    normalized = code.replace("_", ".").strip(".")
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    if not normalized or not is_valid_wb_id(normalized):
        return None
    return normalized


@st.cache_data(show_spinner=False, ttl=24*3600)
def wb_search_indicators(keyword: str, max_pages: int = 2, top: int = 50) -> pd.DataFrame:
    """Vector search qua World Bank Data360 để lấy series mô tả."""
    key = (keyword or "").strip()
    if not key:
        return pd.DataFrame()
    payload = {
        "count": True,
        "select": "series_description/idno, series_description/name, series_description/database_id",
        "search": key,
        "top": int(top or 50),
    }
    try:
        resp = requests.post(
            "https://data360api.worldbank.org/data360/searchv2",
            json=payload,
            headers={**HEADERS, "Content-Type": "application/json"},
            timeout=REQ_TIMEOUT,
        )
        resp.raise_for_status()
        js = resp.json()
    except Exception as exc:
        st.error(f"Lỗi khi tìm chỉ số: {exc}")
        return pd.DataFrame()

    items = js.get("value", []) if isinstance(js, dict) else []
    results = []
    for it in items:
        sd = (it or {}).get("series_description") or {}
        _id = sd.get("idno", "").strip()
        _name = sd.get("name", "").strip()
        _source = sd.get("database_id", "").strip()
        normalized = normalize_indicator_id(_id, _source)
        score = it.get("@search.score")
        if not _id or not _name or not normalized:
            continue
        results.append(
            {
                "id": _id,
                "normalized_id": normalized,
                "name": _name,
                "unit": "",
                "source": _source,
                "search_score": float(score) if isinstance(score, (int, float)) else None,
            }
        )
    if not results:
        return pd.DataFrame(columns=["id", "normalized_id", "name", "unit", "source", "search_score"])
    df = pd.DataFrame(results)
    df["search_score"] = pd.to_numeric(df["search_score"], errors="coerce").fillna(0.0)
    return (
        df.drop_duplicates(subset=["id"])
        .sort_values(["search_score", "name"], ascending=[False, True])
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

st.set_page_config(page_title="World Bank Indicators — Sửa python7", layout="wide")
st.title("Công cụ tổng hợp và phân tích dữ liệu vĩ mô kết hợp AI")
st.caption(" ")

# ===== Sidebar: Tool tìm indicator, chọn năm, Xử lý N/A, Quốc gia =====
with st.sidebar:
    st.header("🔧 Công cụ")
    # Quốc gia
    country_labels = [label for label, _ in COUNTRY_OPTIONS]
    default_country = country_labels[0:1]
    country_choices = st.multiselect(
        "Chọn quốc gia (ISO code)",
        options=country_labels,
        default=default_country,
        help="Có thể chọn nhiều quốc gia, mỗi lựa chọn đã hiển thị kèm mã ISO.",
    )
    # Tìm indicator
    st.subheader("Tìm chỉ số (World Bank)")
    kw = st.text_input("Từ khoá", value="GDP")
    top_n = st.number_input("Top", 1, 500, 10, 1)
    do_search = st.button("🔍 Tìm indicator")

    if do_search:
        if not kw.strip():
            st.warning("Nhập từ khoá trước khi tìm.")
        else:
            with st.spinner("Đang tìm indicators từ World Bank…"):
                df_ind = wb_search_indicators(kw.strip(), max_pages=1, top=int(top_n))
                if top_n:
                    df_ind = df_ind.head(int(top_n))
                st.session_state["ind_search_df"] = df_ind

    # Khoảng năm + xử lý NA
    col_from, col_to = st.columns(2)
    with col_from:
        y_from = st.number_input(
            "Từ năm",
            min_value=1960,
            max_value=2035,
            value=DEFAULT_DATE_RANGE[0],
            step=1,
        )
    with col_to:
        y_to = st.number_input(
            "Đến năm",
            min_value=1960,
            max_value=2035,
            value=DEFAULT_DATE_RANGE[1],
            step=1,
        )
    na_method = st.selectbox(
        "Xử lý chỉ tiêu có dữ liệu N/A",
        [
            "Giữ nguyên (N/A)",
            "Điền 0",
            "Forward-fill theo Country + cột dữ liệu",
            "Backward-fill theo Country + cột dữ liệu",
        ],
        index=0,
    )

selected_country_codes: List[str] = []
for label in country_choices:
    code = COUNTRY_LABEL_TO_CODE.get(label)
    if code:
        selected_country_codes.append(code)

selected_country_codes = [c.upper() for c in selected_country_codes if c]
seen = set()
selected_country_codes = [c for c in selected_country_codes if not (c in seen or seen.add(c))]

# ===== Main area: Tabs riêng biệt =====
TAB_TITLES = ["📊 Dữ liệu", "📈 Biểu đồ", "🧮 Thống kê", "📥 Xuất dữ liệu", "🤖 AI"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_TITLES)

# Tải kết quả tìm kiếm để chọn indicator
ind_df = st.session_state.get("ind_search_df", pd.DataFrame())
if not ind_df.empty and "normalized_id" not in ind_df.columns:
    ind_df["normalized_id"] = ind_df["id"].apply(lambda x: normalize_indicator_id(x))
indicator_df = (
    ind_df.dropna(subset=["normalized_id"])
    if not ind_df.empty and "normalized_id" in ind_df.columns
    else ind_df
)
raw_to_normalized = {
    row["id"]: row.get("normalized_id")
    for _, row in (indicator_df if not indicator_df.empty else pd.DataFrame()).iterrows()
    if row.get("normalized_id")
}
id_to_name = {
    row.get("normalized_id"): row["name"]
    for _, row in (indicator_df if not indicator_df.empty else pd.DataFrame()).iterrows()
    if row.get("normalized_id")
}

with tab1:
    st.subheader("Chọn chỉ số để tải dữ liệu")
    selected_indicator_ids: List[str] = []
    all_indicator_ids = indicator_df["id"].tolist() if not indicator_df.empty else []
    current_state = st.session_state.get("indicator_selection", {})

    if indicator_df.empty:
        st.info("Hãy dùng thanh bên trái để *Tìm indicator*. Toàn bộ chỉ số hợp lệ từ World Bank sẽ được hiển thị tại đây.")
    else:
        display_df = indicator_df[["id", "name", "source"]].copy()
        state_filtered = {row["id"]: current_state.get(row["id"], False) for _, row in indicator_df.iterrows()}
        display_df.insert(0, "Chọn", display_df["id"].map(state_filtered).fillna(False))
        display_df = display_df.rename(columns={"name": "Tên chỉ tiêu", "source": "Nguồn"})
        editor_df = display_df.set_index("id")
        edited_df = st.data_editor(
            editor_df[["Chọn", "Tên chỉ tiêu", "Nguồn"]],
            hide_index=True,
            use_container_width=True,
            height=260,
            column_config={
                "Chọn": st.column_config.CheckboxColumn("Chọn", help="Tick để thêm vào danh sách tải"),
                "Tên chỉ tiêu": st.column_config.Column("Tên chỉ tiêu"),
                "Nguồn": st.column_config.Column("Nguồn"),
            },
        )
        updated_state = {ind_id: bool(row["Chọn"]) for ind_id, row in edited_df.iterrows()}
        st.session_state["indicator_selection"] = updated_state
        selection_mode = st.radio(
            "Phạm vi chỉ tiêu",
            ["Theo lựa chọn", "All chỉ tiêu tìm thấy"],
            horizontal=True,
        )
        if selection_mode == "All chỉ tiêu tìm thấy":
            selected_indicator_ids = all_indicator_ids
        else:
            selected_indicator_ids = [ind_id for ind_id, checked in updated_state.items() if checked]
    use_friendly = True
    load_clicked = st.button(
        "📥 Tải dữ liệu",
        type="primary",
        use_container_width=True,
        disabled=indicator_df.empty,
    )

    if load_clicked:
        if y_from > y_to:
            st.error("Năm bắt đầu phải nhỏ hơn hoặc bằng năm kết thúc.")
            st.stop()
        if not selected_indicator_ids:
            st.warning("Chọn ít nhất một chỉ số (tick hoặc chọn All).")
            st.stop()
        if not selected_country_codes:
            st.warning("Chọn ít nhất một quốc gia ở thanh bên trái.")
            st.stop()
        if "all" in [c.lower() for c in selected_country_codes]:
            country_list = ["all"]
        else:
            country_list = selected_country_codes
        normalized_selection: List[str] = []
        for raw_id in selected_indicator_ids:
            mapped = raw_to_normalized.get(raw_id)
            if mapped:
                normalized_selection.append(mapped)
        chosen_ids = [cid for cid in normalized_selection if cid and is_valid_wb_id(cid)]
        if not chosen_ids:
            st.error("Không có ID hợp lệ sau khi lọc.")
            st.stop()
        ordered_display_columns: List[str] = []
        for cid in chosen_ids:
            col_name = id_to_name.get(cid, cid) if use_friendly else cid
            if col_name not in ordered_display_columns:
                ordered_display_columns.append(col_name)
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
        df_wide = pivot_wide(df_long, use_friendly_name=use_friendly, id_to_name=id_to_name)
        df_wide = handle_na(df_wide, na_method)
        for col in ordered_display_columns:
            if col not in df_wide.columns:
                df_wide[col] = None
        base_cols = ["Country", "Năm"]
        for base in base_cols:
            if base not in df_wide.columns:
                df_wide[base] = None
        base_cols_present = [c for c in base_cols if c in df_wide.columns]
        other_cols = [c for c in df_wide.columns if c not in base_cols_present + ordered_display_columns]
        df_wide = df_wide[base_cols_present + ordered_display_columns + other_cols]
        st.session_state["wb_df_wide"] = df_wide
        st.session_state["chart_defaults"] = [c for c in df_wide.columns if c not in ("Năm", "Country")]
        st.session_state["last_selected_indicator_ids"] = chosen_ids
        st.session_state["last_selected_indicator_names"] = [id_to_name.get(cid, cid) for cid in chosen_ids]
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
        if not value_cols:
            st.info("Không có cột dữ liệu để vẽ.")
        else:
            df_long_plot = df.melt(
                id_vars=["Năm", "Country"],
                value_vars=value_cols,
                var_name="Indicator",
                value_name="Value",
            )
            default_choices = st.session_state.get("chart_defaults", [])
            default_choices = [c for c in default_choices if c in value_cols]
            if not default_choices:
                default_choices = value_cols[:min(4, len(value_cols))]
            choose = st.multiselect(
                "Chọn chỉ số để vẽ",
                options=value_cols,
                default=default_choices,
            )
            if choose:
                st.session_state["chart_defaults"] = choose
                df_plot = df_long_plot[df_long_plot["Indicator"].isin(choose)].copy()
                fig = px.line(
                    df_plot.sort_values(["Country", "Indicator", "Năm"]),
                    x="Năm",
                    y="Value",
                    color="Indicator",
                    line_group="Country",
                    markers=True,
                )
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
        target_audience = st.selectbox("Đối tượng tư vấn", ["Ngân hàng Agribank","Nhân viên Ngân hàng", "Chủ doanh nghiệp"])
        if genai is None or not (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY")):
            st.info("Chưa cấu hình GEMINI_API_KEY nên bỏ qua AI insight.")
        else:
            if st.button("🚀 AI phân tích chỉ tiêu và đề xuất hành động"):
                try:
                    api_key = (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY"))
                    genai.configure(api_key=api_key)
                    model_name = "gemini-2.5-flash"
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