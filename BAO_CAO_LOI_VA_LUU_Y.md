# Lỗi đã gặp và lưu ý khi tiếp tục trên Colab

Tài liệu ghi lại triệu chứng từ các lần chạy thực tế của Trần Đình Duy. Số huấn luyện và chấm điểm chính thức nằm trong `REPORT.md` và sẽ được đối chiếu với `report/evidence/` khi có file.

| Triệu chứng | Điều đã quan sát | Xử lý trong notebook hiện tại |
|---|---|---|
| E4 báo thiếu `e2.json` | E2 chưa chấm, dù có thể đã train | Chạy `infer_and_score("E2")` từ checkpoint đã lưu, rồi mới chạy E4 |
| `Attempting to unscale FP16 gradients` | Kiểm tra optimizer thấy 8 bias projector có gradient FP16 | `apply_setup` chuyển tất cả tham số trainable sang FP32 trước khi tạo optimizer; kiểm tra dtype |
| Projector chỉ đếm khoảng 54,58 triệu tham số | Skip module visual không khớp tên nhánh thực tế | Nạp với `llm_int8_skip_modules=["model.visual", "lm_head"]`; kiểm tra visual không chứa `Linear4bit`. Bản E2 thành công đếm 109.105.152 tham số projector |
| `CheckpointError` khi backward | Kết hợp checkpointing với nhánh visual đóng băng gây số tensor lưu lúc forward và recompute khác nhau | E1/E2 tắt checkpointing ở visual blocks; E3 đảm bảo đầu vào vision có gradient và dùng checkpointing phù hợp |
| `NameError: train_experiment` hoặc `infer_and_score` | Runtime Colab mới không giữ các định nghĩa Python của phiên trước | Chạy lại cell **định nghĩa** ở mục 6/7 cùng các cell phụ thuộc; không chạy lại cell train E2 nếu Drive đã có checkpoint |
| Colab báo hết hạn GPU | Hạn mức GPU tài khoản miễn phí | Không có thời điểm mở lại cố định; giữ checkpoint ở Drive, tiếp tục khi Colab cấp GPU. Không để hai notebook cùng tranh runtime |
| `Run all` xếp hàng train/suy luận lại | Colab đang chạy notebook từ trên xuống | Dừng hàng đợi nếu vừa ấn nhầm, kiểm tra checkpoint và file evidence trước khi chạy riêng cell cần thiết |

## Quy tắc giữ kết quả

- `runs/E2/checkpoint-300` và `runs/E2/run.json` trên Google Drive là **kết quả huấn luyện E2**, chưa phải F1. File `e2.json` chỉ được tạo sau khi suy luận đủ 311 ảnh và chạy trọng tài.
- Output E0 trong notebook cho F1 làm tròn 0,095 nhưng file `e0.json` chưa có trong repository. Không dựng file evidence từ output đã làm tròn.
- Mỗi runtime Colab mới cần nạp lại dữ liệu, model và các định nghĩa hàm. `DRIVE_RUNS` chỉ có sau khi mount Drive.
- Notebook mới tự sao lưu `work/*.jsonl` theo từng batch và `report/evidence/*.json` sau khi chấm vào Drive `day10-vlm/artifacts/`. Kiểm tra file backup trước khi đóng runtime. Evidence JSON được commit; prediction JSONL, ảnh gốc và trọng số không được commit.
- Không sửa `scripts/` hoặc `tests/` của đề bài.
