# Báo cáo Lab Day 10 — Qwen3-VL-4B định vị hư hại mặt đường

**Họ tên:** Trần Đình Duy

**Mã sinh viên:** 2A202602631

**Dataset version:** `d10-21e4070301` (output kiểm tra split đã lưu trong `lab_day10.ipynb`).

> **Trạng thái bằng chứng:** Các số E0 bên dưới được chép từ output chấm điểm lưu trong notebook và đã làm tròn như dòng lệnh in ra. Số E2 về huấn luyện lấy từ output 300 bước của chính notebook. Chưa có các file `report/evidence/e0.json`–`e4*.json` trong repository, nên những ô không có số được ghi **chưa có bằng chứng**. Không dùng báo cáo này để thay thế evidence do `scripts/score.py` sinh ra.

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1@IoU0.5 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | Zero-shot | 0 | Chưa ghi | Chưa ghi | 0,095 | 0,067–0,125 | 0,82 | 0,123 | 0,029 | 0,212 |
| E1 | LoRA LLM | 33.030.144¹ | 10,362413¹ | 44,251577¹ | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm |
| E2 | E1 + mở projector | 142.135.296 | 8,42² | 44,5² | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm |
| E3 | E2 + LoRA ViT | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| E4 r=4 | Ablation `rank`, alpha=8 | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| E4 r=64 | Ablation `rank`, alpha=128 | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |

¹ Metadata E1 lấy từ `run.json` người thực hiện cung cấp: 300 bước, `peak_vram_gb=10.362412929534912`, `train_minutes=44.25157671504997`, loss cuối 0,5561. Hai giá trị hiển thị ở bảng được làm tròn tới 6 chữ số sau dấu phẩy như lệnh tạo evidence trong notebook; cần đối chiếu với `e1.json` sau khi chấm. ² Log E2 ở bước 300 ghi VRAM 8,42 GB và thời gian 44,5 phút; `run.json` trên Drive là nguồn cần đối chiếu lại để điền độ chính xác đầy đủ. Các số E0 đã làm tròn, cần thay bằng giá trị từ `e0.json` trước khi nộp.

## 10 câu hỏi

### Câu 1 — Kiến trúc ViT, projector, LLM và khối được cập nhật

Luồng suy luận là **ảnh → ViT → merger/projector → embedding của LLM → chuỗi JSON**. ViT lấy đặc trưng ảnh, projector đưa đặc trưng đó về chiều ẩn của mô hình ngôn ngữ, rồi LLM sinh nhãn và bốn tọa độ bằng token. Ở mẫu 640×360 đã kiểm tra, `image_grid_thw=[1,22,40]`; gộp không gian 2×2 cho `22×40/4 = 220` token thị giác.

E1 chỉ cập nhật LoRA trên LLM: 252 module LoRA, 33.030.144 tham số trainable theo output chạy. E2 giữ LoRA LLM và mở `visual.merger` cùng `visual.deepstack_merger_list`: notebook đếm **109.105.152** tham số projector và **142.135.296** tham số trainable tổng. E3 dự định thêm LoRA vào attention/MLP của ViT nhưng chưa chạy. Số tham số **toàn bộ** ViT và **toàn bộ** LLM chưa được đo từ mô hình đang dùng, nên chưa thể báo cáo hai con số đó như kết quả của mình. Cần in tổng `numel()` theo tiền tố `model.visual` và `model.language_model` trên model gốc khi có GPU, rồi ghi vào đây.

### Câu 2 — Baseline E0

Output `score.py` đã lưu trong notebook ghi `parse_valid_rate = 0,894` trên **311** ảnh, tương đương khoảng 10,6% đầu ra không được parser công nhận. E0 đạt precision **0,141**, recall **0,072**, F1 **0,095** và CI95 **[0,067; 0,125]**. Hai vấn đề có thể xác nhận là **định dạng không hợp lệ ở một phần đầu ra** và **bỏ sót/định vị sai hư hại** (recall thấp, đặc biệt crack **0,023**). Chưa lưu mẫu raw của đầu ra parse lỗi, nên chưa thể khẳng định kiểu lỗi JSON nào xuất hiện nhiều nhất. Ví dụ raw có thật trong output notebook: `[{"label":"manhole","bbox_2d": [770, 640, 843, 687]}]`; đây là ví dụ đúng cú pháp, không được dùng làm minh chứng lỗi parse. Cần lọc `work/e0.jsonl` bằng `parse_output` để trích hai ví dụ lỗi thực tế khi khôi phục/chạy lại E0.

### Câu 3 — Thang tọa độ 0–1000 và pixel

Nhãn YOLO `[xc,yc,w,h]` trong `[0,1]` được đổi thành `(xc±w/2, yc±h/2)×1000`, làm tròn và chặn về `[0,1000]`. Một box có x trong khoảng 320–480 pixel ở ảnh rộng 640 tương ứng x khoảng 500–750 trên thang 1000. Nếu gửi box 0–1000 cho `score.py --coord pixel`, trọng tài sẽ tiếp tục chia x cho 640 và y cho 360 rồi nhân 1000: box bị phóng/đẩy lệch, IoU và F1 thay đổi. Chiều ngược lại làm box bị co về gần góc trên trái. Kiểm tra trong mã notebook: `objects_to_answer` dùng thang 1000; E0 và các setup mặc định chấm `norm1000`, còn E4 pixel phải dùng `--coord pixel`.

**Chưa có phép chấm đối chứng trên cùng file prediction với hai cờ `--coord`**, nên chưa thể báo F1 sai thang bằng số. Trước khi nộp, chấm cùng `work/e0.jsonl` hoặc một file dự đoán khác với hai cờ `norm1000` và `pixel`, lưu kết quả đối chứng riêng và ghi hai F1 thật vào đây. Không thay thế evidence chính thức bằng lần chấm sai thang.

### Câu 4 — Mask loss trên câu trả lời assistant

Mục tiêu SFT là dự đoán JSON của assistant theo điều kiện ảnh và prompt. Hàm `encode_example` tạo hai chuỗi bằng cùng chat template: chuỗi kết thúc ngay trước câu trả lời (`add_generation_prompt=True`) và chuỗi đầy đủ có câu trả lời. Notebook xác nhận chuỗi prompt là **tiền tố** của chuỗi đầy đủ, sau đó gán `labels[:n_prompt] = -100`. PyTorch bỏ qua các vị trí này khi tính cross entropy; chỉ JSON đáp án và token kết thúc assistant đóng góp vào loss. Khi padding batch, nhãn padding cũng là `-100`, còn `pixel_values` và `image_grid_thw` được nối theo chiều 0.

Kiểm tra mẫu thực tế trong notebook: `image_grid_thw=[1,22,40]`, tương ứng **220 token ảnh** sau merge; giải mã các token có nhãn khác `-100` cho JSON đáp án rồi `<|im_end|>`. Nếu tính loss cả prompt và vị trí token ảnh trong chuỗi, gradient sẽ tối ưu thêm việc dự đoán lại phần điều kiện đã cho thay vì tập trung vào chuỗi box đích. Cách nói “tái tạo ảnh” chỉ là diễn giải trực giác: loss ngôn ngữ áp lên token/placeholder, không trực tiếp hồi quy lại pixel ảnh.

### Câu 5 — Lượng tử hóa, projector và float32

Để E2 cập nhật projector đầy đủ, các lớp `visual.merger` và `visual.deepstack_merger_list` phải giữ trọng số khả vi dạng floating point. Notebook hiện dùng `llm_int8_skip_modules=["model.visual", "lm_head"]` khi nạp QLoRA NF4, nhằm giữ nhánh thị giác, bao gồm projector nằm bên trong `model.visual`, khỏi bị thay bằng lớp BitsAndBytes 4-bit. Trước đó cấu hình skip không khớp tên module đã dẫn tới trường hợp chỉ đếm được khoảng **54,58 triệu** tham số projector, trong khi bản hiện tại đếm **109.105.152**. Lượng tử hóa riêng ViT không phải định luật khiến mọi projector không thể train; vấn đề của mô hình và cấu hình này là projector **cùng nằm trong nhánh visual bị lượng tử hóa**.

Lần chạy E2 cũ lỗi `Attempting to unscale FP16 gradients`; kiểm tra optimizer khi đó thấy **8 bias** projector còn gradient FP16. Mã sửa chuyển **toàn bộ tham số trainable** sang `float32` trước khi tạo optimizer, rồi xác nhận tập dtype trainable chỉ gồm `torch.float32`. Cách này tránh lỗi `GradScaler` và giúp cập nhật projector ổn định hơn trong môi trường tính toán FP16. Output E2 sau sửa cho thấy train đủ 300 bước, loss cuối **0,4835**, VRAM đỉnh **8,42 GB**. Trọng số projector được lưu riêng trong `projector.pt` và phải nạp lại khi suy luận E2/E3.

### Câu 6 — Chia dữ liệu theo phiên quay

Frame liên tiếp của dashcam có gần như cùng cảnh và cùng hư hại. Nếu chia ngẫu nhiên theo ảnh, frame lân cận có thể rơi vào train và test, làm F1 test đo khả năng nhận lại cảnh quen thay vì khái quát sang phiên quay mới. `prepare_data.py` gom theo tiền tố ngày/giờ của tên ảnh, rồi chia tất định theo session. Output kiểm tra trong notebook cho thấy **không có session trùng giữa các split**.

| Split | Ảnh | Session | Pothole | Crack | Manhole | Tổng box |
|---|---:|---:|---:|---:|---:|---:|
| Train | 1.422 | 13 | 1.074 | 1.750 | 691 | 3.515 |
| Val | 276 | 2 | 66 | 470 | 103 | 639 |
| Test | 311 | 3 | 121 | 299 | 163 | 583 |

Tổng là **2.009 ảnh**, **18 session**, **4.737 box**. Split theo session tạo lệch lớp thực tế; ví dụ val chỉ có 66 pothole, nên cần đọc F1 theo lớp và CI95.

### Câu 7 — E1/E2/E3: lợi ích so với tài nguyên

Metadata huấn luyện E1 cho thấy **33.030.144** tham số trainable, **10,362413 GB**, **44,251577 phút** cho 300 bước, loss cuối **0,5561**. Log E2 cho thấy **142.135.296** tham số trainable, **8,42 GB**, **44,5 phút** cho 300 bước, loss cuối **0,4835**. Không nên kết luận E2 tiết kiệm VRAM hơn E1 từ hai số đỉnh này: phải xác nhận hai lần đo dùng cùng điều kiện runtime và cách reset bộ đếm. E1/E2 chưa có F1 và CI95 trong repo; E3 chưa chạy. Vì thiếu kết quả chấm test, **chưa thể kết luận** mở projector hoặc LoRA ViT có đáng chi phí hay không. Sau khi có evidence, so sánh chênh lệch F1 cùng CI95 và chi phí VRAM/phút trên cùng tập test 311 ảnh.

### Câu 8 — CI95 trên 311 ảnh

Trọng tài bootstrap lấy mẫu lại **311 ảnh có hoàn lại**, tính micro F1 cho mỗi mẫu, lặp **1.000** lần với seed cố định và lấy các phân vị 2,5%–97,5%. E0 in ra CI95 **[0,067; 0,125]**, độ rộng **0,058** (các số đã làm tròn); F1 điểm là **0,095**. Khoảng này thể hiện độ nhạy của kết quả E0 với thành phần ảnh test, nhất là khi chỉ có ba session test. Cùng một tập ảnh còn tạo phụ thuộc giữa các setup, nên chỉ nhìn CI95 từng setup có giao nhau hay không là dấu hiệu thô. Để khẳng định chênh lệch nhỏ, nên bootstrap **hiệu F1 theo cặp trên cùng ảnh**, tốt hơn nữa xem thêm biến thiên theo session. Chưa có evidence E1–E3 nên chưa thể tính độ rộng CI hoặc khẳng định setup nào hơn setup nào.

### Câu 9 — Lỗi theo lớp và ảnh minh họa

Ở E0, **crack yếu nhất**: P **0,038**, R **0,023**, F1 **0,029**; pothole F1 **0,123**, manhole F1 **0,212**. Test có **299** box crack, nhiều hơn pothole (**121**), vì vậy không thể quy kết kết quả E0 thấp đơn thuần do thiếu mẫu test. Giả thuyết kỹ thuật là crack thường mảnh/dài, ranh giới khó xác định và IoU@0.5 nhạy với lệch tọa độ; cần xem box thực tế mới xác nhận nguyên nhân cho các ảnh cụ thể. **Chưa có sáu ảnh dự đoán đối chiếu GT** vì dữ liệu ảnh và `work/e0.jsonl`/`work/e1.jsonl` chưa được xuất về local. Mục này phải bổ sung ảnh thật, chú thích TP/FP/FN và liên kết tới file ảnh trước khi nộp.

### Câu 10 — E4 ablation đơn biến

Notebook đã chuẩn bị E4 cho **biến rank LoRA**, hai mức `r=4, alpha=8` và `r=64, alpha=128`. Cùng một setup nền tốt nhất E1–E3, tập train, số bước, learning rate, prompt, thang tọa độ và tập test phải giữ nguyên. Alpha thay đổi theo quy tắc `alpha=2r`, giữ tỉ số alpha/r bằng 2; đây là cách thực thi biến độ rộng rank theo đề bài. **Hai nhánh chưa chạy**, chưa có `e4_rank_4.json` hay `e4_rank_64.json`, nên chưa thể nêu chiều thay đổi F1 hay kết luận độ tin cậy. Sau khi chạy, báo F1 và CI95 của cả hai; CI giao nhau không chứng minh hai setup bằng nhau, và CI không giao nhau là bằng chứng mạnh hơn nhưng vẫn cần xét paired bootstrap khi so sánh sát nhau.

## Nhìn lại và việc còn thiếu trước khi nộp

Các lỗi thực tế đã gặp: gradient FP16 của 8 bias projector làm `GradScaler` dừng E2; sai cấu hình skip visual khiến projector không được mở đầy đủ; `CheckpointError` khi kết hợp gradient checkpointing với nhánh thị giác; phiên Colab mất GPU trước khi hoàn thiện inference/evidence. E2 hiện có checkpoint-100/200/300 và `run.json` trên Drive theo ảnh người thực hiện cung cấp. Bản notebook cần được chạy theo từng cell và lưu kết quả về Drive/GitHub; output hiển thị trong giao diện Colab không thay thế file evidence. Các mục cần làm tiếp: khôi phục/chạy lại E0 và E1 để lấy evidence, chấm E2 từ checkpoint-300, train/chấm E3, chạy cả hai nhánh E4, tạo sáu ảnh minh họa và đối chiếu toàn bộ số trong bảng với evidence trước khi nộp.
