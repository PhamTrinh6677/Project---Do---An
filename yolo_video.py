
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# Import các thư viện cần thiết
import numpy as np
import argparse
import imutils
import time
import cv2
import os


# Xây dựng các phân tích đối số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
        help="path to input video")
ap.add_argument("-o", "--output", required=True,
        help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
        help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Gắn nhãn lớp COCO mô hình YOLO đã được đào tạo
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Khởi tạo danh sách các màu để biểu diễn cho các nhãn
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

# Lấy đường dẫn các trọng số YOLO và cấu hình mô hình
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load trình phát hiện đối tượng YOLO được đào tạo về bộ dữ liệu YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#  Khởi tạo luồng video , con trỏ đến tệp video đầu ra và kích thước khung hình
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Cố gắng để xác định tổng số khung hình trong tệp video
try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} Frames trong Video".format(total))

# Thông báo lỗi
except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

# Lặp các khung hình từ tệp video đầu vào
d=1
while True:
        
        # Đọc khung tiếp theo từ tệp 
        (grabbed, frame) = vs.read()    

        # Nếu khung hình không được lấy nữa nghĩa là đã đến luồng cuối
        if not grabbed:
                break

        # Nếu kích thước khung trống sẽ gán giá trị H,W từ khung thật trong video
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        # Xây dựng blod từ khung đầu vào và sau đó thực hiện đưa vào máy dò đối tượng YOLO
	# Cho chúng ta các hộp giới hạn của chúng ta và xác suất ,liên quan
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # Khởi tạo danh sách các hộp giới hạn , điểm tin cậy , ID lớp
        boxes = []
        confidences = []
        classIDs = []

        # Lặp qua mỗi đầu ra của lớp
        for output in layerOutputs:
                # Lặp qua từng phát hiện
                for detection in output:
                        # Trích suất ID lớp , điểm tin cậy của phát hiện đối tượng hiện tại
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        #  Lọc các dự đoán yếu 
                        if confidence > args["confidence"]:
                                #  YOLO trả về tọa độ trung tâm (x,y) và chiều rộng và chiều cao của các ô
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                #  sử dụng tọa độ trung tâm (x,y) để lấy được góc trên bên trái của hộp giới hạn
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # Cập nhật danh sách các tọa độ hộp giới hạn , điểm tin cậy , ID lớp
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # Áp dụng non-maxima để loại trừ các hộp giới hạn kém
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        # Tạo luồng video nếu luồng video chưa có
        if writer is None:
                
                
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                        (frame.shape[1], frame.shape[0]), True)
                if total > 0 :
                        elap = (end - start)
                        print("[INFO] Một khung hình sử dụng {:.2f} giây".format(elap))
                        print("[INFO] Ước tính tổng thời gian hoàn thành {:.2f} phút".format(
				(elap * total)/60))
        # Đảm bảo tồn tại ít nhất một phát hiện       
        if len(idxs) > 0:
                
                for i in idxs.flatten():
                        # Tọa độ của hộp giới hạn
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # Kiểm tra các khung hình là người
                        if(LABELS[classIDs[i]]=='person' or LABELS[classIDs[i]]=='knife' or LABELS[classIDs[i]]=='scissors') :
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}".format(LABELS[classIDs[i]])
                                text1 = "Frame {} : {} {},{},{},{}".format(d,LABELS[classIDs[i]],x,y,w,h)
                                file = open("Care.txt","a")
                                file.write(text1+"\n")
                                file.close()
                                #if(LABELS[classIDs[i]]=='person') :
                                cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        '''if(LABELS[classIDs[i]]=='knife' or LABELS[classIDs[i]]=='scissors') :
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}".format(LABELS[classIDs[i]])
                                text1 = "Frame {} : Knife {},{},{},{}".format(d,x,y,w,h)
                                file = open("Care.txt","a")
                                file.write(text1+"\n")
                                file.close()
                                cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        '''
                           
                for i in idxs.flatten():
                        if(LABELS[classIDs[i]]=='person') :
                                writer.write(frame)
                                break
        d = d + 1  


print("[INFO] cleaning up...")
file = open("Care.txt","a")
file.write("===================================================================\n")
file.close()
writer.release()
vs.release()
