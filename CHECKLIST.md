# Checklist trước khi nộp — Trần Đình Duy (2A202602631)

Cập nhật 15/10/2026. Chỉ đánh dấu hoàn tất khi file thật có trong repository và số trong `REPORT.md` khớp evidence.

| Việc | Trạng thái | Bằng chứng / thao tác tiếp theo |
|---|---|---|
| Repository cá nhân có ít nhất 5 commit | Đạt | Đã vượt ngưỡng 5 commit; kiểm tra số mới nhất bằng `git rev-list --count HEAD` |
| Không track dữ liệu, khóa và trọng số | Đạt | `git ls-files` và `check_submission.py` đều không thấy file cấm |
| Split theo session đã kiểm tra | Đạt | `data/splits/meta.json` từ Colab khớp `d10-21e4070301` và 1.422/276/311 ảnh; file nằm trong `data/` bị Git bỏ qua |
| `e0.json` | Có trong repo local | `report/evidence/e0.json`: 311 ảnh, F1 0,095346, CI95 [0,066736; 0,124861] |
| `e1.json` | Có trong repo local | `report/evidence/e1.json`: parse 0,993569, F1 0,262626, hơn E0 0,167280 |
| `e2.json` | Có trong repo local | `report/evidence/e2.json`: parse 1,0, F1 0,324910; metadata khớp `report/training/e2_run.json` |
| `e3.json` | Có trong repo local | `report/evidence/e3.json`: 311 ảnh, parse 1,0, F1 0,435411, CI95 [0,393443; 0,480148] |
| E4 rank r=4 và r=64 | Đủ hai evidence | r=4: F1 0,455516, CI95 [0,416523; 0,495528]; r=64: F1 0,448339, CI95 [0,408747; 0,489152]. Hai CI giao nhau |
| Sáu ảnh GT/prediction | Đạt | `report/examples/` có 3 ảnh có TP và 3 ảnh lỗi, kèm README và chú thích từng ảnh trong Câu 9 |
| `REPORT.md` | Đã cập nhật E0–E4 | Bảng, 10 câu, đối chứng sai thang tọa độ, phân tích theo session và 6 ảnh thật; chưa ghi E5 khi chưa có evidence |
| `check_submission.py` | 0 FAIL, 0 WARN trên máy local | `data/splits/meta.json` trong zip khớp evidence; file này bị Git bỏ qua. Bản clone sạch không có dữ liệu sẽ hiện 1 WARN cho phép đối chiếu metadata |
| `python -m pytest -q` | 67 passed, 1 skipped, 1 failed | Test mẫu trong `tests/test_configs_and_docs.py` đòi `REPORT.md` còn `<điền`, trái với điều kiện nộp đã điền; không sửa file test của giảng viên |
| Repository GitHub Public | Đạt | Trang GitHub hiển thị `Public` khi mở không đăng nhập |
| Nộp URL trên LMS | Chưa làm | Nộp liên kết repo Public sau khi kiểm tra nội dung đã đẩy; E5 chỉ bổ sung nếu kịp hạn |

## Evidence đã lấy từ Colab

Bốn file `report/evidence/e0.json`–`e3.json` cùng `e4_rank_4.json` và `e4_rank_64.json` đã được đặt đúng vị trí trong repo. Zip từ Colab đã cung cấp `work/e3.jsonl`, `data/splits/meta.json`, `test.jsonl` và sáu ảnh test đã vẽ; dữ liệu `work/`, `data/` và zip nằm ngoài danh sách commit. Chấm lại E3 bằng trọng tài tái tạo đúng toàn bộ số trong `e3.json`. Nếu muốn phân tích sâu thêm E0 hoặc bootstrap hiệu F1 ghép cặp, còn cần prediction JSONL tương ứng từ Drive; đó không phải điều kiện bắt buộc của hồ sơ E0–E4.

Notebook sao lưu `work/*.jsonl` sau mỗi batch suy luận và `report/evidence/*.json` sau khi chấm vào `/content/drive/MyDrive/day10-vlm/artifacts/` của tài khoản Colab đang dùng. Sau khi mount đúng Drive và chạy cell định nghĩa `restore_artifact`, có thể khôi phục một file đã sao lưu bằng `restore_artifact("work/e3.jsonl")`. Evidence E0–E3 trong repo không chứa các dự đoán theo từng ảnh.

## Sau khi hoàn thành E4 trên GPU Colab

E0–E3 và cả hai nhánh E4 đã train/chấm xong. **Không bấm Run all** hay chạy lại E1–E4. Giữ checkpoint trên Drive để đối chiếu metadata; E5 là phần thưởng tùy chọn, chỉ bổ sung khi có kết quả thật.

```python
from pathlib import Path
DRIVE_RUNS = Path("/content/drive/MyDrive/day10-vlm-new/runs")
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
print("Nơi lưu checkpoint E4:", DRIVE_RUNS)
```

Hàm `_resolve_run` tìm `run.json` và checkpoint ở `DRIVE_RUNS` nếu không còn bản local. Nếu runtime phải khởi động lại để làm E5, cần chuẩn bị lại `data/splits/test.jsonl`, ảnh test và các cell định nghĩa; không cần train lại E0–E4.

### Phiên GPU mới: chỉ chạy E5

Chạy từng cell theo thứ tự dưới đây, dựa vào dòng đầu của cell trong `lab_day10.ipynb`. Không dùng **Run all**.

1. Cell `# Colab starts in /content; clone...` để khôi phục repo (bỏ qua cell `%cd` đứng trước nếu repo chưa tồn tại); tiếp theo là cell `# Runtime → Change runtime type → T4 GPU` để cài requirements, rồi cell `# Mount Drive...`.
2. Cell `# Kaggle: dùng Colab Secrets...`, `!python scripts/prepare_data.py --kaggle`, rồi cell `# Verify the fixed session split...` để tái tạo `data/` sau khi runtime bị xóa.
3. Chạy các cell **định nghĩa** bắt đầu bằng `# Convert YOLO labels...`, `# Encode one image...`, `# Freeze the base model...`, `# Train one of E1/E2/E3...`, `# Reload the base model...`. Cell train chỉ định nghĩa hàm; không tự gọi `train_experiment`.
4. Chọn đúng Drive chứa checkpoint **E3** trước khi chạy E5:

```python
import json
from pathlib import Path

DRIVE_RUNS = Path("/content/drive/MyDrive/day10-vlm-new/runs")
e3_meta = DRIVE_RUNS / "E3" / "run.json"
assert e3_meta.is_file(), f"Không thấy checkpoint E3: {e3_meta}"
e3_run = json.loads(e3_meta.read_text(encoding="utf-8"))
e3_checkpoint = DRIVE_RUNS / "E3" / Path(e3_run["checkpoint"]).name
assert (e3_checkpoint / "adapter").is_dir(), e3_checkpoint
assert (e3_checkpoint / "projector.pt").is_file(), e3_checkpoint
print("Sẽ khởi tạo E5 từ:", e3_checkpoint)
```

5. Kiểm tra `data/splits/meta.json`, `data/splits/train.jsonl`, `data/splits/test.jsonl` đều tồn tại. Sau đó **chỉ chạy** cell `# E5: verifiable IoU reward...`. Cell này huấn luyện và chấm E5; bốn generation với Qwen3-VL-4B có thể vượt VRAM T4, nên theo dõi bộ nhớ và không coi evidence là hoàn tất trước khi có `report/evidence/e5.json`.

Không chạy lại cell E0, các cell gọi `train_experiment("E1"/"E2"/"E3")`, cell `infer_and_score` riêng, hoặc cell E4 rank ablation. Nếu notebook mở trước khi GitHub cập nhật, thêm `from trl import GRPOConfig, GRPOTrainer` vào cell E5 hoặc mở lại bản mới từ GitHub; không restart runtime chỉ để lấy một dòng import.

Sau mỗi lần chấm, sao lưu cả `report/evidence/*.json` và `work/*.jsonl` vào Drive hoặc tải về máy. Chỉ đưa evidence JSON và ảnh minh họa đã chọn lên GitHub; `work/`, `data/`, khóa và checkpoint phải ở ngoài Git. Khi đã có dữ liệu/prediction ở local, tạo sáu ảnh thật bằng:

```powershell
python report/visualize_examples.py --pred work/e3.jsonl
```

Script đã được dùng để vẽ ba ảnh có true positive và ba ảnh có FP/FN vào `report/examples/`, kèm `README.md`; ảnh đã được nhúng ở Câu 9 của báo cáo.

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

Trước khi nộp, đối chiếu từng ô báo cáo với `report/evidence/*.json`, xác nhận repository Public và chỉ nộp khi `check_submission.py` báo **0 FAIL**. E5 có thể bổ sung sau bằng evidence/log thật nếu còn thời gian và hạn nộp cho phép.
