import cv2
import threading
from ultralytics import YOLO
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 初始化 VideoCapture 对象
URL = "rtsp://admindaniel:921213@192.168.11.68:554/stream1"
cap = cv2.VideoCapture(URL)

NUM = 0
model = YOLO(r"D:\Car dection\plate.pt")

def show_video():
    # 視窗名稱
    window_name = "daniel cardetction"
    # 創建視窗
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 視窗大小
    cv2.resizeWindow(window_name, 1280, 720)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # 車牌辨識
            results = model(frame, conf=0.05)
            res = results[0]
            boxes_list = res.boxes
            bbox_points_list = []

            #處理目標檢测，使用Tesseract OCR 辨識文本，然後在圖像上繪製
            for box in boxes_list.xyxy:
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                bbox_points_list.append([x1, y1, x2, y2])
                crop_img = frame[y1:y2, x1:x2]
                cur_text = pytesseract.image_to_string(crop_img, config='--psm 11')
                cur_text = re.sub(r'[^\w\d]', '', cur_text)#正規表達式去除所有非字母和數字字符
                text_color = (200, 200, 200)
                background_color = (210, 0, 111)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(cur_text, font, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 + baseline - 10), background_color, -1)
                cv2.putText(frame, cur_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 3)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# 直接调用 show_video 函数
show_video()
