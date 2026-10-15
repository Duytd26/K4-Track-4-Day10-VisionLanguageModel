# Sáu ví dụ thật từ E3 trên tập test

Ảnh gốc 640 × 360 thuộc split `test`, version `d10-21e4070301`. Xanh lá: ground truth; đỏ: dự đoán E3 từ `work/e3.jsonl`. TP/FP/FN tính theo cùng lớp và IoU ≥ 0,5. Nhóm `correct` nghĩa là có ít nhất một TP, không nhất thiết không có FN.

- [correct_01_vlcsnap-2025-02-18-23h10m48s405.png](correct_01_vlcsnap-2025-02-18-23h10m48s405.png): TP=3, FP=0, FN=0; đúng crack, manhole, pothole.
- [correct_02_vlcsnap-2025-02-19-17h04m47s362.png](correct_02_vlcsnap-2025-02-19-17h04m47s362.png): TP=3, FP=0, FN=0; đúng ba pothole.
- [correct_03_vlcsnap-2025-02-18-23h29m04s331.png](correct_03_vlcsnap-2025-02-18-23h29m04s331.png): TP=3, FP=0, FN=1; đúng ba crack, sót một crack.
- [error_01_vlcsnap-2025-02-19-17h05m37s430.png](error_01_vlcsnap-2025-02-19-17h05m37s430.png): TP=0, FP=5, FN=4; năm box manhole chồng lấn tại vùng pothole, nhầm lớp và sót GT.
- [error_02_vlcsnap-2025-02-18-23h29m07s392.png](error_02_vlcsnap-2025-02-18-23h29m07s392.png): TP=0, FP=4, FN=4; nhãn crack đúng nhưng box lệch, không đạt IoU 0,5.
- [error_03_vlcsnap-2025-02-18-23h30m28s990.png](error_03_vlcsnap-2025-02-18-23h30m28s990.png): TP=0, FP=1, FN=7; một box crack quá rộng, bỏ sót bốn pothole và ba crack.

Tất cả sáu output đều parse JSON hợp lệ. Chú thích chi tiết và ảnh nhúng nằm trong [`REPORT.md`](../../REPORT.md).
