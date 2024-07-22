import cv2
import threading
from ultralytics import YOLO
import pytesseract
import re
import torch
import pandas as pd

torch.cuda.empty_cache()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 初始化 VideoCapture 对象
URL = "rtsp://admindaniel:921213@192.168.11.68:554/stream1"
cap = cv2.VideoCapture(URL)

NUM = 0
model = YOLO(r"D:\Car dection\plate.pt")

#存車牌
license_plates = []
# 緩存已識別
known_plates = set()

def save_to_excel(plates):
    df = pd.DataFrame(plates, columns=['License Plate'])
    df.to_excel('license_plates.xlsx', index=False)
    print("車牌正在保存 license_plates.xlsx")

def show_video():
    # 视窗名称
    window_name = "daniel cardetction"
    # 创建视窗
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 视窗大小
    cv2.resizeWindow(window_name, 1280, 720)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # 車牌識別
            results = model(frame, conf=0.05)
            res = results[0]
            boxes_list = res.boxes
            bbox_points_list = []

            # 使用Tesseract OCR 識別，然後繪製
            new_plates_detected = False
            for box in boxes_list.xyxy:
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                bbox_points_list.append([x1, y1, x2, y2])
                crop_img = frame[y1:y2, x1:x2]
                cur_text = pytesseract.image_to_string(crop_img, config='--psm 11')
                cur_text = re.sub(r'[^\w\d]', '', cur_text)  # 正则表达式去除所有非字母和数字字符
                
                if cur_text and cur_text not in known_plates:
                    known_plates.add(cur_text)
                    license_plates.append(cur_text)
                    # 保存Excel文件
                    save_to_excel(license_plates)
                    new_plates_detected = True

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
            
            # 如果新的车牌，暂停識別，直到下一個車牌出现
            if new_plates_detected:
                print("dected new car_plate，暂停識別中...")
                while True:
                    # 在此暂停識別
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, conf=0.05)
                    res = results[0]
                    boxes_list = res.boxes
                    
                    # 檢查是否有新的車牌出现
                    new_plate_found = False
                    for box in boxes_list.xyxy:
                        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                        crop_img = frame[y1:y2, x1:x2]
                        cur_text = pytesseract.image_to_string(crop_img, config='--psm 11')
                        cur_text = re.sub(r'[^\w\d]', '', cur_text)
                        
                        if cur_text and cur_text not in known_plates:
                            new_plate_found = True
                            break
                    
                    if new_plate_found:
                        break
                    cv2.waitKey(5000)  # 每秒检查一次

    cap.release()
    cv2.destroyAllWindows()

# 直接调用 show_video 函数
show_video()
