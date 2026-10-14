# Checklist trước khi nộp — Trần Đình Duy (2A202602631)

Cập nhật 14/10/2026. Chỉ đánh dấu hoàn tất khi file thật có trong repository và số trong `REPORT.md` khớp evidence.

| Việc | Trạng thái | Bằng chứng / thao tác tiếp theo |
|---|---|---|
| Repository cá nhân có ít nhất 5 commit | Đạt | Đã vượt ngưỡng 5 commit; kiểm tra số mới nhất bằng `git rev-list --count HEAD` |
| Không track dữ liệu, khóa và trọng số | Đạt | `git ls-files` và `check_submission.py` đều không thấy file cấm |
| Split theo session đã kiểm tra | Đạt trong output notebook | `d10-21e4070301`, 1.422/276/311 ảnh; cần có `data/splits/meta.json` khi đối chiếu cuối |
| `e0.json` | Có trong repo local | `report/evidence/e0.json`: 311 ảnh, F1 0,095346, CI95 [0,066736; 0,124861] |
| `e1.json` | Có trong repo local | `report/evidence/e1.json`: parse 0,993569, F1 0,262626, hơn E0 0,167280 |
| `e2.json` | Có trong repo local | `report/evidence/e2.json`: parse 1,0, F1 0,324910; metadata khớp `report/training/e2_run.json` |
| `e3.json` | Chưa chạy | Train E3, sau đó suy luận và chấm |
| E4 rank r=4 và r=64 | Chưa chạy | Chỉ chạy sau khi có `e1.json`–`e3.json`; notebook tự chọn setup nền theo F1 |
| Sáu ảnh GT/prediction | Thiếu | Cần dữ liệu ảnh và prediction JSONL thật; ghi rõ TP/FP/FN, không dùng ảnh minh họa giả |
| `REPORT.md` | Đã cập nhật E0–E2, còn thiếu E3/E4 và ảnh | Bảng và phân tích E0–E2 lấy từ evidence thật; không điền số giả cho E3/E4 |
| `check_submission.py` | 2 FAIL, 1 WARN | E0/E1/E2 và metadata đã PASS; thiếu `e3.json`, `e4*.json`. Chưa có `data/splits/meta.json` local nên chưa đối chiếu được dataset version với split |
| Repository GitHub Public | Đạt | Trang GitHub hiển thị `Public` khi mở không đăng nhập |
| Nộp URL trên LMS | Chưa làm | Chỉ thực hiện khi đã có evidence và kiểm tra cuối đạt 0 FAIL |

## Evidence đã lấy từ Colab

Ba file `report/evidence/e0.json`, `e1.json`, `e2.json` đã được đặt đúng vị trí trong repo local. Vẫn cần lấy `work/e0.jsonl` và `work/e1.jsonl` (hoặc `work/e2.jsonl`) cùng ảnh test từ Colab/Drive để phân tích lỗi và vẽ sáu ảnh thật. Chỉ commit evidence và ảnh minh họa đã chọn; không commit toàn bộ `work/` hay `data/`.

Notebook sao lưu `work/*.jsonl` sau mỗi batch suy luận và `report/evidence/*.json` sau khi chấm vào `/content/drive/MyDrive/day10-vlm/artifacts/` của tài khoản Colab đang dùng. Sau khi mount đúng Drive và chạy cell định nghĩa `restore_artifact`, có thể khôi phục một file đã sao lưu bằng `restore_artifact("work/e1.jsonl")`. Evidence E0–E2 trong repo không chứa các dự đoán theo từng ảnh.

## Tiếp tục E3 khi có GPU Colab

Mở notebook từ GitHub, chọn T4. **Chạy từng cell**, không chọn Run all. Chạy cell clone repo (mục 0), cài thư viện, mount Drive, nạp Kaggle Secrets và chuẩn bị dữ liệu/split; sau đó chạy cell định nghĩa nhãn/prompt (mục 2), chat template/dataset (mục 4), `apply_setup` (mục 5), `train_experiment` (mục 6) và **cell chỉ định nghĩa** `infer_and_score` (mục 7). E1/E2 đã có evidence; **không gọi lại** `train_experiment("E1")` hoặc `train_experiment("E2")`. Nếu `DRIVE_RUNS` đang trỏ vào lối tắt thư mục `runs` chỉ có quyền Viewer, đổi nó sang một thư mục riêng có quyền ghi trên Drive tài khoản mới trước khi train E3:

```python
from pathlib import Path
DRIVE_RUNS = Path("/content/drive/MyDrive/day10-vlm-new/runs")
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
run_info_e3 = train_experiment("E3")
evidence_e3 = infer_and_score("E3", evidence_name="E3")
```

Hàm `_resolve_run` tìm `run.json` và checkpoint ở `DRIVE_RUNS` nếu không còn bản local. Việc chuẩn bị lại `data/splits/test.jsonl` và ảnh test vẫn bắt buộc; checkpoint không chứa dữ liệu. Nếu `infer_and_score` báo `NameError`, cell định nghĩa hàm hoặc hàm phụ thuộc chưa chạy trong runtime hiện tại. Chạy E4 chỉ sau khi E3 đã được chấm và `report/evidence/e3.json` có trong runtime.

Sau mỗi lần chấm, sao lưu cả `report/evidence/*.json` và `work/*.jsonl` vào Drive hoặc tải về máy. Chỉ đưa evidence JSON và ảnh minh họa đã chọn lên GitHub; `work/`, `data/`, khóa và checkpoint phải ở ngoài Git. Khi đã có dữ liệu/prediction ở local, tạo sáu ảnh thật bằng:

```powershell
python report/visualize_examples.py --pred work/e1.jsonl
```

Script vẽ ba ảnh có true positive và ba ảnh có FP/FN vào `report/examples/`, kèm `README.md` mô tả từng ảnh. Sau đó thêm liên kết các ảnh vào Câu 9 của báo cáo.

## Kiểm tra cuối trên máy có đủ evidence

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/check_submission.py
git status --short
git ls-files data work kaggle.json .env '*.pt' '*.safetensors' '*.bin'
git rev-list --count HEAD
```

Trước khi nộp, đối chiếu từng ô báo cáo với `report/evidence/*.json`, thêm sáu ảnh thật, xác nhận repository Public và chỉ nộp khi `check_submission.py` báo **0 FAIL**.
