from PyQt5 import QtCore 
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem,QHeaderView
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtCore import QTimer
import time
import os
import numpy as np
from ultralytics import YOLO
import easyocr
import pytesseract
import re
from ui import Ui_MainWindow
from img_controller import img_controller
from datetime import datetime

import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
    num_digits = len(digits_and_letters)
    print(f"Number of digits: {num_digits}")
    return ''.join(digits_and_letters)


def process_image(image_path,label,plates,running_time,tableWidget):
    start_time = time.time()
    frame = cv2.imread(image_path)
    print(image_path)
    if frame is None:
        print(f"Error: Unable to read image from {image_path}")
    car_model = YOLO(r"D:\Car dection\best.pt")
    word_model = YOLO(r"D:\Car dection\best_word.pt")
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

        show_crop_on_label(label,crop_img)

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
        print(detected_plates)
        detected_plates_str = "\n".join([" ".join(plate) for plate in detected_plates])
        plates.setText(f"車牌號碼:{detected_plates_str}")

    tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tableWidget.horizontalHeader().setMinimumSectionSize(100)

    end_time = time.time()
    running_time_e = end_time - start_time
    running_time.setText(f"Processing Time: {running_time_e:.2f}")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_position = tableWidget.rowCount()
    tableWidget.insertRow(row_position)

    tableWidget.setItem(row_position, 0, QTableWidgetItem(current_time))  # 当前时间
    tableWidget.setItem(row_position, 1, QTableWidgetItem(final_text))

    # 插入图像
    rgb_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    height, width, channel = rgb_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image).scaled(100, 50, QtCore.Qt.KeepAspectRatio)
    image_item = QTableWidgetItem()
    image_item.setData(QtCore.Qt.DecorationRole, pixmap)
    tableWidget.setItem(row_position, 2, image_item)

    # 插入运行时间
    tableWidget.setItem(row_position, 3, QTableWidgetItem(f"{running_time_e:.2f} 秒"))

    # 在插入数据后调整列宽
    tableWidget.resizeColumnsToContents()
    tableWidget.resizeRowsToContents()
    tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    tableWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    tableWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
    tableWidget.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
    

def show_crop_on_label(plate_image,crop_img): 

    rgb_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    height, width, channel = rgb_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image)
    plate_image.setPixmap(pixmap)

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.file_path = ''  
        self.setup_control()
        self.setup_expert_camera()

    def setup_expert_camera(self):
        self.expert_cap = None
        self.expert_timer = QTimer(self)
        self.expert_timer.timeout.connect(self.update_expert_frame)
        self.ui.connect.clicked.connect(self.connect_expert_camera)
    
    def connect_expert_camera(self):
        ip = self.ui.ip.toPlainText()
        port = self.ui.ip.toPlainText()
        rtsp_url = f"rtsp://admin:123456@{ip}:{port}/stream1"  # 根据您的摄像机RTSP URL格式进行调整
        self.expert_cap = cv2.VideoCapture(rtsp_url)
        if self.expert_cap.isOpened():
            self.expert_timer.start(30)  # 每30ms更新一次帧
            print("Connected to Expert IP Camera")
        else:
            print("Failed to connect to Expert IP Camera")

    def update_expert_frame(self):
        if self.expert_cap:
            ret, frame = self.expert_cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.ui.label_2.setPixmap(pixmap)
                self.ui.label_2.setScaledContents(True)

    def closeEvent(self, event):
        if self.expert_cap:
            self.expert_cap.release()
        event.accept()


    def setup_control(self):
        self.img_controller = img_controller(img_path=self.file_path,
                                             label_img=self.ui.label_img,
                                             label_file_name=self.ui.label_file_name,
                                             label_ratio=self.ui.label__radio,
                                             label_img_shape=self.ui.label_img_shape)
        self.setup_expert_camera()
        self.ui.btn_open_file.clicked.connect(self.open_file)         
        self.ui.btn_zoom_in.clicked.connect(self.img_controller.set_zoom_in)
        self.ui.btn_zoom_out.clicked.connect(self.img_controller.set_zoom_out)
        self.ui.silder_zoom.valueChanged.connect(self.getslidervalue)
        self.ui.btn_execute.clicked.connect(self.execute)
        self.ui.delete_list.clicked.connect(self.delete)
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./") # start path        
        if filename:
            self.file_path = filename  
            self.init_new_picture(filename)

    def init_new_picture(self, filename):
        self.ui.silder_zoom.setProperty("value", 50)
        self.img_controller.set_path(filename)

    def getslidervalue(self):        
        self.img_controller.set_slider_value(self.ui.silder_zoom.value()+1)
    
    def execute(self):
        if self.file_path:
            print(f"Processing image: {self.file_path}")
            process_image(self.file_path, self.ui.plate_image,self.ui.plates,self.ui.running_time,self.ui.tableWidget)  # Pass QLabel to process_image
        else:
            print("No file selected.")
    def delete(self):
        rowCount = self.ui.tableWidget.rowCount()
        for row in range(rowCount):
            self.ui.tableWidget.removeRow(0)

    

        

  

    