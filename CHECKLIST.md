# Checklist trước khi nộp — Trần Đình Duy (2A202602631)

Cập nhật 14/10/2026. Chỉ đánh dấu hoàn tất khi file thật có trong repository và số trong `REPORT.md` khớp evidence.

| Việc | Trạng thái | Bằng chứng / thao tác tiếp theo |
|---|---|---|
| Repository cá nhân có ít nhất 5 commit | Đạt | 13 commit khi rà soát |
| Không track dữ liệu, khóa và trọng số | Đạt | `git ls-files` và `check_submission.py` đều không thấy file cấm |
| Split theo session đã kiểm tra | Đạt trong output notebook | `d10-21e4070301`, 1.422/276/311 ảnh; cần có `data/splits/meta.json` khi đối chiếu cuối |
| `e0.json` | Thiếu trong repo | Nếu runtime Colab cũ còn sống, lấy `report/evidence/e0.json`; nếu không, chạy lại cell E0 rồi `score.py` |
| `e1.json` | Thiếu trong repo | Đã nhận `run.json` E1 với 300 bước, VRAM 10,362413 GB, 44,251577 phút; cần checkpoint/adapter trên Drive để chạy `infer_and_score("E1")`, hoặc xuất evidence gốc từ runtime cũ |
| `e2.json` | Chưa chấm | Đã nhận `run.json` E2 với 300 bước, VRAM 8,419290 GB, 44,454236 phút; checkpoint ở Drive `runs/E2/checkpoint-300`. Dựng lại môi trường rồi chạy `infer_and_score("E2")` |
| `e3.json` | Chưa chạy | Train E3, sau đó suy luận và chấm |
| E4 rank r=4 và r=64 | Chưa chạy | Chỉ chạy sau khi có `e1.json`–`e3.json`; notebook tự chọn setup nền theo F1 |
| Sáu ảnh GT/prediction | Thiếu | Cần dữ liệu ảnh và prediction JSONL thật; ghi rõ TP/FP/FN, không dùng ảnh minh họa giả |
| `REPORT.md` | Đã điền phần có căn cứ, còn thiếu số | Cập nhật bảng/câu 1–3, 7–10 theo evidence thật; kiểm tra các ghi chú `Chưa...` |
| `check_submission.py` | Chưa đạt | Sau khi điền report: 5 FAIL, 1 WARN do thiếu evidence và meta local; chạy lại sau khi import evidence |
| Repository GitHub Public | Đạt | Trang GitHub hiển thị `Public` khi mở không đăng nhập |
| Nộp URL trên LMS | Chưa làm | Chỉ thực hiện khi đã có evidence và kiểm tra cuối đạt 0 FAIL |

## Lấy `e0.json` từ Colab

Trong **đúng runtime đã chạy E0**, mở thanh Files (biểu tượng thư mục), tìm `/content/K4-Track-4-Day10-VisionLanguageModel/report/evidence/e0.json`, bấm menu ba chấm → **Download**. Tải thêm `work/e0.jsonl` để phân tích lỗi và vẽ ảnh. Tải file evidence về máy, đặt đúng `report/evidence/e0.json` trong repository. Nếu runtime đã bị xóa và file chưa sao lưu, output hiển thị trong notebook không thể khôi phục mọi trường trong JSON; phải chạy lại suy luận/chấm E0.

## Tiếp tục E2 khi có GPU Colab

Mở notebook từ GitHub, chọn T4. **Chạy từng cell**, không chọn Run all. Chạy cell clone repo (mục 0), cài thư viện, mount Drive, nạp Kaggle Secrets và chuẩn bị dữ liệu/split; sau đó chạy cell định nghĩa nhãn/prompt (mục 2), chat template/dataset (mục 4), `apply_setup` (mục 5), `train_experiment` (mục 6) và **cell chỉ định nghĩa** `infer_and_score` (mục 7). Với E2 đã có checkpoint, **không gọi lại** `train_experiment("E2")`. Gọi riêng:

```python
evidence_e2 = infer_and_score("E2", evidence_name="E2")
print(evidence_e2["micro"]["f1"])
```

Hàm `_resolve_run` tìm `run.json` và checkpoint ở `DRIVE_RUNS` nếu không còn bản local. Việc chuẩn bị lại `data/splits/test.jsonl` và ảnh test vẫn bắt buộc; checkpoint không chứa dữ liệu. Nếu `infer_and_score` báo `NameError`, cell định nghĩa hàm hoặc hàm phụ thuộc chưa chạy trong runtime hiện tại.

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
