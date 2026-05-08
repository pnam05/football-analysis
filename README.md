# Football Video Analysis System

Hệ thống ứng dụng Computer Vision và Deep Learning để phân tích video các trận đấu bóng đá. Pipeline này thực hiện việc phát hiện, theo dõi đối tượng (cầu thủ, trọng tài, quả bóng), phân loại đội bóng, tính toán tỷ lệ kiểm soát bóng, cũng như ước lượng tốc độ và quãng đường di chuyển của từng cầu thủ trong không gian thực.

## Demo Hệ Thống

## [![Watch Demo](https://youtu.be/v8cuC_662gA/0.jpg)](https://youtu.be/v8cuC_662gA)

## Các tính năng chính

- **Phát hiện và Theo dõi đối tượng (Detection & Tracking)**: Sử dụng mô hình YOLO kết hợp với thuật toán ByteTrack (thông qua thư viện supervision) để nhận diện và theo dõi vị trí của cầu thủ, trọng tài và quả bóng xuyên suốt video một cách liền mạch.

- **Phân loại Đội bóng (Team Assignment)**: Trích xuất màu áo cầu thủ và sử dụng thuật toán K-Means Clustering để tự động phân chia cầu thủ về hai đội khác nhau.

- **Xử lý Hậu Tracking (Post-processing)**: Tích hợp module xử lý đứt gãy track và tự động sửa lỗi gán nhầm ID (ID Swaps) dựa trên khoảng cách vật lý và dự đoán đội bóng, giúp luồng tracking ổn định hơn.

- **Ước lượng Chuyển động Camera (Camera Movement Estimation)**: Cấu hình Optical Flow (Lucas-Kanade) để tính toán độ dời của camera, từ đó điều chỉnh lại tọa độ chuẩn cho các object trên sân.

- **Chuyển đổi Góc nhìn (Perspective Transformation)**: Áp dụng ma trận biến đổi góc nhìn (Bird's-eye view) để map tọa độ pixel (2D) sang tọa độ thực tế trên sân (mét).

- **Phân tích Thể lực & Tốc độ (Speed & Distance)**: Tính toán vận tốc (km/h) và tổng quãng đường di chuyển (m) của từng cầu thủ.

- **Thống kê Kiểm soát bóng (Ball Possession)**: Gán quyền sở hữu bóng cho cầu thủ gần nhất và thống kê tỷ lệ kiểm soát bóng theo thời gian thực (Team Ball Control %).

## Cấu trúc dự án

├── input_video/ # Thư mục chứa video đầu vào
├── output_video/ # Thư mục chứa video kết quả sau khi render
├── stubs/ # Nơi lưu trữ file pickle (cache tracking/camera movement để tiết kiệm thời gian chạy lại)
├── camera_movement_estimator.py # Tính toán và bù trừ chuyển động của camera (Optical Flow)
├── main.py # File thực thi chính, kết nối toàn bộ pipeline
├── player_ball_assigner.py # Logic gán quả bóng cho cầu thủ đang kiểm soát
├── post_processor.py # Xử lý nội suy, vá lỗi mất track và ID swap
├── speed_and_distance_estimator.py # Tính toán quãng đường và tốc độ dựa trên tọa độ thực
├── team_assigner.py # Phân cụm cầu thủ theo đội dựa trên màu sắc (KMeans)
├── tracker.py # Khởi tạo YOLO, ByteTrack và các hàm vẽ Bounding Box/Đồ họa
├── view_transformer.py # Chuyển đổi tọa độ Pixel -> Tọa độ sân đấu thực
└── utils.py # Các hàm tiện ích (đọc/ghi video, tính khoảng cách...)

## Công nghệ sử dụng

- **Ngôn ngữ**: Python

- **Deep Learning / Vision**: ultralytics (YOLO), OpenCV (cv2)

- **Tracking**: supervision (ByteTrack)

- **Machine Learning / Data**: scikit-learn (KMeans), numpy, pandas

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

Sử dụng pip để cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### 2. Trọng số mô hình (Weights)

Tải file trọng số YOLO của bạn (ví dụ: best.pt đã được train riêng cho tập dữ liệu bóng đá) và đặt vào thư mục gốc của project.

### 3. Khởi chạy Pipeline

- Đặt video cần phân tích vào thư mục input_video/. (Trong code mặc định đang đọc input_video/08fd33_4.mp4).

- Chạy lệnh sau:

```bash
python main.py
```

- **Lưu ý về Stubs (Cache)**: Trong lần chạy đầu tiên, hệ thống sẽ tiến hành tracking và lưu kết quả vào thư mục stubs/ (đuôi .pkl). Ở những lần chạy sau, hệ thống sẽ đọc trực tiếp từ stubs để tăng tốc quá trình gỡ lỗi (debugging). Nếu bạn đổi video hoặc đổi model, hãy xóa các file trong thư mục **stubs/** để hệ thống tính toán lại từ đầu.

### 4. Kết quả đầu ra

Video sau khi đã vẽ các annotation (vòng elip dưới chân cầu thủ, tam giác trên đầu người giữ bóng, vận tốc, tổng quãng đường và bảng thống kê tỷ lệ giữ bóng) sẽ được lưu tại output_video/output.avi.
