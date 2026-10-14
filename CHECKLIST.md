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
| `e3.json` | Có trong repo local | `report/evidence/e3.json`: 311 ảnh, parse 1,0, F1 0,435411, CI95 [0,393443; 0,480148] |
| E4 rank r=4 và r=64 | Đã có r=4, r=64 đang chạy | `report/evidence/e4_rank_4.json`: F1 0,455516, CI95 [0,416523; 0,495528]; chờ `e4_rank_64.json` để kết luận ablation |
| Sáu ảnh GT/prediction | Thiếu | Cần dữ liệu ảnh và prediction JSONL thật; ghi rõ TP/FP/FN, không dùng ảnh minh họa giả |
| `REPORT.md` | Đã cập nhật E0–E3 và E4 r=4 | Còn thiếu E4 r=64 và sáu ảnh; không kết luận ablation khi chưa đủ hai nhánh |
| `check_submission.py` | 0 FAIL, 1 WARN | Một file E4 đủ qua điều kiện tự động; rubric phần B vẫn cần cả hai giá trị. Chưa có `data/splits/meta.json` local để đối chiếu split |
| Repository GitHub Public | Đạt | Trang GitHub hiển thị `Public` khi mở không đăng nhập |
| Nộp URL trên LMS | Chưa làm | Chỉ thực hiện khi đã có evidence và kiểm tra cuối đạt 0 FAIL |

## Evidence đã lấy từ Colab

Bốn file `report/evidence/e0.json`–`e3.json` và `e4_rank_4.json` đã được đặt đúng vị trí trong repo local. Vẫn cần lấy `work/e0.jsonl` và một file dự đoán đã finetune (ưu tiên `work/e3.jsonl`) cùng ảnh test từ Colab/Drive để phân tích lỗi và vẽ sáu ảnh thật. Chỉ commit evidence và ảnh minh họa đã chọn; không commit toàn bộ `work/` hay `data/`.

Notebook sao lưu `work/*.jsonl` sau mỗi batch suy luận và `report/evidence/*.json` sau khi chấm vào `/content/drive/MyDrive/day10-vlm/artifacts/` của tài khoản Colab đang dùng. Sau khi mount đúng Drive và chạy cell định nghĩa `restore_artifact`, có thể khôi phục một file đã sao lưu bằng `restore_artifact("work/e3.jsonl")`. Evidence E0–E3 trong repo không chứa các dự đoán theo từng ảnh.

## Tiếp tục E4 trên GPU Colab

E3 đã train/chấm xong; cell E4 đang chạy và đã chọn E3 làm setup nền. **Không bấm Run all** hay chạy lại E1–E3. Giữ Drive lưu run mới ở thư mục tài khoản hiện tại có quyền ghi, rồi đợi E4 hoàn tất cả `r=4` và `r=64`. Nếu runtime mất kết nối giữa hai nhánh, kiểm tra file evidence và checkpoint trên Drive trước khi chạy lại riêng nhánh còn thiếu.

```python
from pathlib import Path
DRIVE_RUNS = Path("/content/drive/MyDrive/day10-vlm-new/runs")
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
print("Nơi lưu checkpoint E4:", DRIVE_RUNS)
```

Hàm `_resolve_run` tìm `run.json` và checkpoint ở `DRIVE_RUNS` nếu không còn bản local. Khi E4 đang chạy, không đổi `DRIVE_RUNS` giữa chừng. Nếu runtime phải khởi động lại, cần chuẩn bị lại `data/splits/test.jsonl`, ảnh test và chạy lại các cell định nghĩa trước khi chấm nhánh còn thiếu.

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
