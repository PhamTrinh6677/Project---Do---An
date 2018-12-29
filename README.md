# Project---Do---An
Cấu trúc file sẽ như sau 
Trong thư mục tổng tên yolo-object-detection
có 4 phần chính
- Thư mục Videos chứa video đầu vào
- Thư mục Output chứa video đầu ra
- File yolo-video.py
- Thư mục yolo-coco chứa 3 phần cần tải từ Internet là yolov3.cfg , yolov3.weight , coco.name
Sử dụng câu lệnh trong cmd 
" python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco "
Note , Nó chỉ chạy trong OpenCV 4.0.0+ VÀ python 3.7+
