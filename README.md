# Streamlit — Chương trình thu thập & tổng hợp thông tin vĩ mô VN

## Cách chạy nhanh
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tính năng chính
- Lấy dữ liệu World Bank cho nhiều chỉ số cùng lúc (2000–2024 mặc định).
- Xử lý thiếu dữ liệu: giữ nguyên / ffill-bfill / điền mean / điền median.
- Biểu đồ: Line, Bar, Scatter, Combo + Heatmap tương quan.
- Thống kê mô tả: Mean, Std, Min/Max & năm, Median, Q1/Q3, Hệ số biến thiên.
- Tải CSV dữ liệu đã xử lý.
- **AI phân tích & tư vấn**: sinh 5 phần theo yêu cầu (bối cảnh, xu hướng, tương quan, kiến nghị cho Agribank, hành động + KPI).
  - Nhập `OPENAI_API_KEY` trong **Sidebar** hoặc đặt biến môi trường.