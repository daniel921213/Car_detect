import cv2
from ultralytics import YOLO
import pytesseract
import re
import torch
import numpy as np
torch.cuda.empty_cache()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 初始化 VideoCapture 对象
URL = "rtsp://admindaniel:921213@192.168.11.68:554/stream1"
cap = cv2.VideoCapture(URL)

NUM = 0
model = YOLO(r"D:\Car dection\best.pt")

# 创建背景减法器对象
fgbg = cv2.createBackgroundSubtractorMOG2()

def show_video():
    # 視窗名稱
    window_name = "daniel cardetction"
    plate_window_name = "detected plates"
    snapshot_window_name = "snapshot"
    # 創建視窗
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  

    cv2.namedWindow(plate_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(plate_window_name, 640, 360)

    cv2.namedWindow(snapshot_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(snapshot_window_name, 640, 360)

    vehicle_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景减法
        fgmask = fgbg.apply(frame)
        # 去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 找到轮廓
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 检测车辆进入
        vehicle_present = False
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # 调整这个面积阈值以检测车辆进入
                vehicle_present = True
                break

        if vehicle_present and not vehicle_detected:
            vehicle_detected = True
            print("Vehicle detected")

            # 拍照保存图像
            image_filename = f"vehicle_{NUM}.jpg"
            cv2.imwrite(image_filename, frame)
            # NUM += 1

            # 显示拍到的照片
            snapshot_img = cv2.imread(image_filename)
            cv2.imshow(snapshot_window_name, snapshot_img)

            # 車牌辨識
            results = model(frame, conf=0.05)
            res = results[0]
            boxes_list = res.boxes
            bbox_points_list = []

            for box in boxes_list.xyxy:
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                bbox_points_list.append([x1, y1, x2, y2])
                crop_img = frame[y1:y2, x1:x2]
                cur_text = pytesseract.image_to_string(crop_img, config='--psm 11')
                cur_text = re.sub(r'[^\w\d]', '', cur_text)  # 正規表達式去除所有非字母和數字字符
                text_color = (200, 200, 200)
                background_color = (210, 0, 111)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(cur_text, font, font_scale, thickness)
                cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 + baseline - 10), background_color, -1)
                cv2.putText(frame, cur_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 3)

                # 在新窗口中顯示檢測到的車牌文字
                if cur_text:
                    plate_img = 255 * np.ones((360, 640, 3), dtype=np.uint8)
                    cv2.putText(plate_img, cur_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.imshow(plate_window_name, plate_img)

        elif not vehicle_present:
            vehicle_detected = False

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 直接调用 show_video 函数
show_video()
