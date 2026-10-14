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
| E4 rank r=4 và r=64 | Đủ hai evidence | r=4: F1 0,455516, CI95 [0,416523; 0,495528]; r=64: F1 0,448339, CI95 [0,408747; 0,489152]. Hai CI giao nhau |
| Sáu ảnh GT/prediction | Thiếu | Cần dữ liệu ảnh và prediction JSONL thật; ghi rõ TP/FP/FN, không dùng ảnh minh họa giả |
| `REPORT.md` | Đã cập nhật E0–E4 | Bảng và Câu 10 đã có số hai nhánh; còn thiếu sáu ảnh minh họa thật và một số phân tích cần prediction theo ảnh |
| `check_submission.py` | 0 FAIL, 1 WARN | Hai file E4 được nhận; local chưa có `data/splits/meta.json` để đối chiếu split. Khi dữ liệu được chuẩn bị, chạy lại để đạt 0 WARN |
| `python -m pytest -q` | 67 passed, 1 skipped, 1 failed | Test mẫu trong `tests/test_configs_and_docs.py` đòi `REPORT.md` còn `<điền`, trái với điều kiện nộp đã điền; không sửa file test của giảng viên |
| Repository GitHub Public | Đạt | Trang GitHub hiển thị `Public` khi mở không đăng nhập |
| Nộp URL trên LMS | Chưa làm | Chỉ thực hiện khi đã có evidence và kiểm tra cuối đạt 0 FAIL |

## Evidence đã lấy từ Colab

Bốn file `report/evidence/e0.json`–`e3.json` cùng `e4_rank_4.json` và `e4_rank_64.json` đã được đặt đúng vị trí trong repo local. Vẫn cần lấy `work/e0.jsonl` và một file dự đoán đã finetune (ưu tiên `work/e3.jsonl`) cùng ảnh test từ Colab/Drive để phân tích lỗi và vẽ sáu ảnh thật. Chỉ commit evidence và ảnh minh họa đã chọn; không commit toàn bộ `work/` hay `data/`.

Notebook sao lưu `work/*.jsonl` sau mỗi batch suy luận và `report/evidence/*.json` sau khi chấm vào `/content/drive/MyDrive/day10-vlm/artifacts/` của tài khoản Colab đang dùng. Sau khi mount đúng Drive và chạy cell định nghĩa `restore_artifact`, có thể khôi phục một file đã sao lưu bằng `restore_artifact("work/e3.jsonl")`. Evidence E0–E3 trong repo không chứa các dự đoán theo từng ảnh.

## Sau khi hoàn thành E4 trên GPU Colab

E0–E3 và cả hai nhánh E4 đã train/chấm xong. **Không bấm Run all** hay chạy lại E1–E4. Giữ checkpoint trên Drive để đối chiếu metadata; E5 là phần thưởng tùy chọn, sẽ chạy ở phiên GPU sau nếu có điều kiện.

```python
from pathlib import Path
DRIVE_RUNS = Path("/content/drive/MyDrive/day10-vlm-new/runs")
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
print("Nơi lưu checkpoint E4:", DRIVE_RUNS)
```

Hàm `_resolve_run` tìm `run.json` và checkpoint ở `DRIVE_RUNS` nếu không còn bản local. Nếu runtime phải khởi động lại để làm E5, cần chuẩn bị lại `data/splits/test.jsonl`, ảnh test và các cell định nghĩa; không cần train lại E0–E4.

Sau mỗi lần chấm, sao lưu cả `report/evidence/*.json` và `work/*.jsonl` vào Drive hoặc tải về máy. Chỉ đưa evidence JSON và ảnh minh họa đã chọn lên GitHub; `work/`, `data/`, khóa và checkpoint phải ở ngoài Git. Khi đã có dữ liệu/prediction ở local, tạo sáu ảnh thật bằng:

```powershell
python report/visualize_examples.py --pred work/e1.jsonl
```

Script vẽ ba ảnh có true positive và ba ảnh có FP/FN vào `report/examples/`, kèm `README.md` mô tả từng ảnh. Sau đó thêm liên kết các ảnh vào Câu 9 của báo cáo.

Nếu Colab vẫn còn runtime chứa ảnh test và `work/e3.jsonl`, chạy một cell dưới đây tại thư mục gốc repo để tạo **một zip nhỏ** phục vụ hoàn thiện báo cáo. Zip gồm sáu ảnh minh họa, các prediction JSONL hiện có và metadata/split test; không chứa toàn bộ dataset, checkpoint hay khóa API. Tải zip về máy và đặt trong workspace để bổ sung Câu 2, 3, 8, 9.

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess, sys
from google.colab import files

assert Path("work/e3.jsonl").is_file(), "Cần khôi phục work/e3.jsonl từ Drive trước"
subprocess.run([sys.executable, "report/visualize_examples.py", "--pred", "work/e3.jsonl"], check=True)
paths = [Path("data/splits/meta.json"), Path("data/splits/test.jsonl")]
paths += sorted(Path("work").glob("e*.jsonl"))
paths += sorted(p for p in Path("report/examples").iterdir() if p.is_file())
archive = Path("/content/day10_review.zip")
with ZipFile(archive, "w", ZIP_DEFLATED) as zf:
    for path in paths:
        if path.is_file():
            zf.write(path, path.as_posix())
print("Các file đã đóng gói:", [str(p) for p in paths if p.is_file()])
files.download(str(archive))
```

## Kiểm tra cuối trên máy có đủ evidence

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/check_submission.py
git status --short
git ls-files data work kaggle.json .env '*.pt' '*.safetensors' '*.bin'
git rev-list --count HEAD
```

Trước khi nộp, đối chiếu từng ô báo cáo với `report/evidence/*.json`, thêm sáu ảnh thật, xác nhận repository Public và chỉ nộp khi `check_submission.py` báo **0 FAIL**.
