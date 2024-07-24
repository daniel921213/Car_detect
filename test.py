import cv2
from ultralytics import YOLO
import pytesseract
import re
import torch
import numpy as np

torch.cuda.empty_cache()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 初始化YOLO模型
model = YOLO(r"D:\Car dection\best.pt")

def preprocess_image(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 圖像銳化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(binary, -1, kernel)
    return sharp

def process_image(image_path):
    # 讀取圖片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    # 車牌辨識
    results = model(frame, conf=0.05)
    res = results[0]
    boxes_list = res.boxes
    detected_plates = []

    # 處理目標檢測，使用Tesseract OCR 辨識文本
    for box in boxes_list.xyxy:
        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
        crop_img = frame[y1:y2, x1:x2]
        processed_img = preprocess_image(crop_img)
        cur_text = pytesseract.image_to_string(processed_img, config='--psm 11')
        cur_text = re.sub(r'[^\w\d]', '', cur_text)  # 正規表達式去除所有非字母和數字字符
        detected_plates.append(cur_text)
        
        text_color = (200, 200, 200)
        background_color = (210, 0, 111)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(cur_text, font, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 + baseline - 10), background_color, -1)
        cv2.putText(frame, cur_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 3)

    # 顯示結果
    window_name = "daniel cardetction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, frame)
    # 創建一個白色背景的圖像來顯示車牌號碼
    plate_window_name = "Detected License Plates"
    plate_frame = np.ones((500, 800, 3), np.uint8) * 255  # 創建一個白色背景的圖像
    y_offset = 30
    for i, plate in enumerate(detected_plates):
        cv2.putText(plate_frame, f"Plate {i + 1}: {plate}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        y_offset += 40

    cv2.namedWindow(plate_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(plate_window_name, 800, 500)
    cv2.imshow(plate_window_name, plate_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 替換這裡的路徑為你要處理的圖片路徑
image_path = r"D:\Car dection\dateset\train\images\LINE_ALBUM_Car_240723_11_jpg.rf.375aaa58629e24ddb6e9826f2bd6f8f5.jpg"
process_image(image_path)
