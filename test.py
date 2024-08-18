import cv2
import re
import time
import numpy as np
from ultralytics import YOLO
import easyocr
import pytesseract

from PyQt5 import QtWidgets, QtGui, QtCore
from ui import Ui_MainWindow

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 文字辨識
def recognize_text_with_model(model, crop_image):
    results = model(crop_image)
    xlist = {}
    chars = model.names
    
    # 每個字符的bbox
    for cid, cbox in enumerate(results[0].boxes):
        # xlist 字典中記錄每個字元的 X 軸座標
        xlist.update({cbox.xyxy[0][0].item(): cid})
    
    # 按 x 座標排序並拼接
    recog_plate = ""
    for x in sorted(xlist.keys()):
        recog_plate += chars[int(results[0].boxes[xlist[x]].cls.item())]
    
    return recog_plate

def more_fore(text):
    digits_and_letters = re.findall(r'\d|[A-Z]', text)
    print(digits_and_letters)
    num_digits = len(digits_and_letters)
    
    print(f"Text: '{text}'")
    print(f"Digits found: {digits_and_letters}")
    print(f"Number of digits: {num_digits}")
    return ''.join(digits_and_letters)

def process_image(image_path):
    start_time = time.time()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to read image from {image_path}")
        

    car_model = YOLO(r"D:\Car dection\best.pt")
    word_model = YOLO(r"D:\Car dection\best_word.pt")

    # 車牌識別
    results = car_model(frame, conf=0.05)
    res = results[0]
    boxes_list = res.boxes
    detected_plates = []
    
    reader = easyocr.Reader(['en'])

    # 切出車牌
    for idx, box in enumerate(boxes_list.xyxy):
        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
        print(x1,y1,x2,y2)

        crop_img = frame[y1:y2, x1:x2]

        crop_window_name = f"Crop Image {idx + 1}"
        cv2.imshow(crop_window_name, crop_img)
        crop_img_filename = f"crop_image_{idx + 1}.jpg"
        cv2.imwrite(crop_img_filename, crop_img)   

        # 這邊使用easyocr來做，因為用自己對付畸形車牌來用在正正方方的車牌發現會有一些錯誤
        ocr_results = reader.readtext(crop_img)
        cur_text = ''.join([res[1] for res in ocr_results])
        cur_text = re.sub(r'[^\w\d]', '', cur_text)
        
        # 自己定義的模型:用來應付畸形車牌
        cur_text_model = recognize_text_with_model(word_model, crop_img)
        cur_text_model = re.sub(r'[^\w\d]', '', cur_text_model)
        print(f"cur_text (OCR): {cur_text}")
        print(f"cur_text_model (Custom Model): {cur_text_model}")
        cur_text_len = more_fore(cur_text)
        cur_text_model_len = more_fore(cur_text_model)

        print(len(cur_text_len))
        print(len(cur_text_model_len))
        # 用了兩個應該夠狠了八，比較哪個比較長就是辨識比較多嘛，準確嗎....?看起來還行
        if len(cur_text_len) >len(cur_text_model_len):
            final_text = cur_text_len
        else:
            final_text = cur_text_model_len

        detected_plates.append(final_text)

        # 繪製結果
        process_window_name = f"Process Image {idx + 1}"
        cv2.imshow(process_window_name, crop_img)

        text_color = (200, 200, 200)
        background_color = (210, 0, 111)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(cur_text, font, font_scale, thickness)

        cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 3)
        cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1 + baseline - 10), background_color, -1)
        cv2.putText(frame, cur_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    end_time = time.time()
    running_time = end_time - start_time
    running_time_text = f"Processing Time: {running_time:.2f} seconds"

    cv2.putText(frame, running_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    window_name = "Detected License Plates"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)
    cv2.imshow(window_name, frame)

    plate_window_name = "Detected License Plates Numbers"
    plate_frame = np.ones((500, 800, 3), np.uint8) * 255 
    y_offset = 30
    for i, plate in enumerate(detected_plates):
        cv2.putText(plate_frame, f"Plate {i + 1}: {plate}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        y_offset += 40

    cv2.namedWindow(plate_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(plate_window_name, 800, 500)
    cv2.imshow(plate_window_name, plate_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = r"D:\下載\Car_dection.v6i.yolov8\train\images\car_99.jpg"
process_image(image_path)
