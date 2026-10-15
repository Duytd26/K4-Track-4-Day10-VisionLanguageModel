# Lỗi đã gặp và lưu ý khi tiếp tục trên Colab

Tài liệu ghi lại triệu chứng từ các lần chạy thực tế của Trần Đình Duy. Số huấn luyện và chấm điểm chính thức nằm trong `REPORT.md` và `report/evidence/` (E0–E3 cùng hai nhánh E4).

| Triệu chứng | Điều đã quan sát | Xử lý trong notebook hiện tại |
|---|---|---|
| E4 báo thiếu `e2.json` | Khi đó E2 chưa chấm, dù đã train | E0–E3 hiện đã có evidence; cell E4 đã chọn E3 làm setup nền |
| `Attempting to unscale FP16 gradients` | Kiểm tra optimizer thấy 8 bias projector có gradient FP16 | `apply_setup` chuyển tất cả tham số trainable sang FP32 trước khi tạo optimizer; kiểm tra dtype |
| Projector chỉ đếm khoảng 54,58 triệu tham số | Skip module visual không khớp tên nhánh thực tế | Nạp với `llm_int8_skip_modules=["model.visual", "lm_head"]`; kiểm tra visual không chứa `Linear4bit`. Bản E2 thành công đếm 109.105.152 tham số projector |
| `CheckpointError` khi backward | Kết hợp checkpointing với nhánh visual đóng băng gây số tensor lưu lúc forward và recompute khác nhau | E1/E2 tắt checkpointing ở visual blocks; E3 đảm bảo đầu vào vision có gradient và dùng checkpointing phù hợp |
| `NameError: train_experiment` hoặc `infer_and_score` | Runtime Colab mới không giữ các định nghĩa Python của phiên trước | Chạy lại cell **định nghĩa** ở mục 6/7 cùng các cell phụ thuộc; không chạy lại cell train E2 nếu Drive đã có checkpoint |
| Colab báo hết hạn GPU | Hạn mức GPU tài khoản miễn phí | Không có thời điểm mở lại cố định; giữ checkpoint ở Drive, tiếp tục khi Colab cấp GPU. Không để hai notebook cùng tranh runtime |
| `Run all` xếp hàng train/suy luận lại | Colab đang chạy notebook từ trên xuống | Dừng hàng đợi nếu vừa ấn nhầm, kiểm tra checkpoint và file evidence trước khi chạy riêng cell cần thiết |
| E3 lỗi `Found an incompatible version of torchao ... 0.10.0` | PEFT 0.21.2 phát hiện `torchao` có sẵn trong Colab nhưng yêu cầu tối thiểu 0.16.0 | Gỡ `torchao` cũ; notebook đã thêm lệnh gỡ trong cell môi trường. Restart runtime nếu Python vẫn thấy package, sau đó chạy lại cell định nghĩa và E3, không train lại E0–E2 |
| `pytest -q` còn 1 FAIL sau khi điền báo cáo | `tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions` yêu cầu REPORT chứa chuỗi `<điền`, trong khi `check_submission.py` yêu cầu chuỗi đó biến mất | Đây là hai điều kiện mâu thuẫn giữa kiểm thử mẫu và nghiệm thu bài đã điền. Giữ REPORT đã điền, không sửa `scripts/` hay `tests/`; kết quả hiện tại là 67 passed, 1 skipped, 1 failed cho test mẫu, còn `check_submission.py` đạt 0 FAIL |
| Runtime mới báo thiếu repo và `work/e3.jsonl` | `/content` bị xóa khi Colab cấp runtime mới; `%cd` chạy trước cell clone | Chạy cell clone trước, mount đúng Drive, khôi phục prediction JSONL từ `day10-vlm/artifacts/work/`, rồi tải/chia lại dữ liệu. Không cần train lại E3 |

## Quy tắc giữ kết quả

- `runs/E2/checkpoint-300` và `runs/E2/run.json` trên Google Drive là **kết quả huấn luyện E2**; F1 thật hiện ở `report/evidence/e2.json` sau khi suy luận đủ 311 ảnh và chấm bằng trọng tài.
- E0–E3 và cả hai nhánh E4 đều đã có evidence JSON thật trong repository. Sáu ảnh GT/dự đoán E3 đã được đưa vào `report/examples/` và nhúng trong Câu 9. Không dựng evidence E5 từ output làm tròn nếu chạy bonus sau.
- Mỗi runtime Colab mới cần nạp lại dữ liệu, model và các định nghĩa hàm. `DRIVE_RUNS` chỉ có sau khi mount Drive.
- Notebook tự sao lưu `work/*.jsonl` theo từng batch và `report/evidence/*.json` sau khi chấm vào Drive `day10-vlm/artifacts/`. `work/e3.jsonl` đã được lấy về local để đối chiếu: chấm lại thang `norm1000` cho F1 **0,435411**, còn cố tình chấm như `pixel` cho F1 **0,000000**. Chỉ commit evidence JSON và sáu ảnh minh họa đã chọn; prediction JSONL, ảnh gốc, zip chuyển giao và trọng số không được commit.
- Không sửa `scripts/` hoặc `tests/` của đề bài.
