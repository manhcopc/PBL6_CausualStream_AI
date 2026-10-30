# Chạy mô hình thật trên macOS (Apple Silicon)

Tài liệu này dùng cho dịch vụ AI trong thư mục này. Backend kết nối tới AI qua `AI_SERVICE_URL`; frontend không gọi AI trực tiếp. Chế độ mặc định trong `.env.example` là mock để kiểm thử luồng khi chưa có checkpoint.

## 1. Chuẩn bị

- Máy Mac Apple Silicon, Python 3.11 và đủ RAM/dung lượng cho mô hình.
- Bản export Full075 được cấp riêng. Không đưa checkpoint lên Git hoặc đóng vào Docker image.
- Giải nén toàn bộ bản export vào `PBL6_Full075_Test_077/model075/`. Thư mục này phải chứa `full-manifest.json` và các tệp trọng số được manifest tham chiếu.

Từ thư mục `ai`:

```bash
python3.11 -m venv .venv-mps
source .venv-mps/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,real]'
cp .env.example .env
```

Trong `.env`, đặt:

```dotenv
INFERENCE_PROVIDER=real
MODEL_CHECKPOINT_PATH=PBL6_Full075_Test_077/model075
MODEL_VERSION=full075-epoch2
```

Những biến còn lại có thể giữ theo `.env.example`. Runtime tự dùng MPS và kiểm tra manifest; chỉ điều chỉnh chu kỳ lấy mẫu sau khi đã chạy được cấu hình cơ bản.

## 2. Chạy dịch vụ

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Khi chạy chế độ thật, dịch vụ kiểm tra manifest và tải mô hình lúc khởi động; bước này có thể mất thời gian. Kiểm tra `http://127.0.0.1:8001/healthz` và `http://127.0.0.1:8001/readyz`, sau đó khởi động Backend với `AI_SERVICE_URL=http://127.0.0.1:8001`.

Nếu cần thử riêng một video, dùng trang phân tích video của frontend khi cả ba dịch vụ đã chạy. Không cần tệp `demo.mp4` trong repo.

## 3. Ghi chú vận hành

- Chỉ chạy một tiến trình tải mô hình trên máy có bộ nhớ hạn chế.
- Chế độ `real` cần checkpoint đầy đủ; nếu thiếu, chuyển `INFERENCE_PROVIDER=mock` để kiểm thử tích hợp.
- Dockerfile hiện tại phục vụ mock trên Linux. MPS chỉ chạy trên macOS và không dùng Dockerfile đó để tải mô hình thật.
- Kết quả kiểm thử mô hình có thể thay đổi theo máy, độ dài video và cấu hình lấy mẫu; nên đo độ trễ trên đúng máy triển khai.
