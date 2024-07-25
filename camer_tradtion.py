from flask import Flask
import cv2
import threading
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = Flask(__name__)

# 初始化 VideoCapture 对象
URL = "rtsp://admindaniel:921213@192.168.11.68:554/stream1"
cap = cv2.VideoCapture(URL)
#把車牌作優化
def preprocess_license_plate(license_plate):

    gray_license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

    equalized_license_plate = cv2.equalizeHist(gray_license_plate)

    smoothed_license_plate = cv2.GaussianBlur(equalized_license_plate, (5, 5), 0)

    _, binary_license_plate = cv2.threshold(smoothed_license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_license_plate

def character_segmentation(binary_license_plate):
    # 二值化后的形態學操作，去除噪声
    kernel = np.ones((3, 3), np.uint8)
    morphed_plate = cv2.morphologyEx(binary_license_plate, cv2.MORPH_CLOSE, kernel)
    
    # 水平投影，去除上下白邊
    horizontal_projection = np.sum(morphed_plate, axis=1)
    top, bottom = 0, morphed_plate.shape[0]
    for row in range(morphed_plate.shape[0]):
        if horizontal_projection[row] > 0:
            top = row
            break
    for row in range(morphed_plate.shape[0]-1, -1, -1):
        if horizontal_projection[row] > 0:
            bottom = row
            break
    cropped_plate = morphed_plate[top:bottom, :]

    # 垂直投影，分割每個字
    vertical_projection = np.sum(cropped_plate, axis=0)
    character_boundaries = []
    in_character = False
    for col in range(cropped_plate.shape[1]):
        if not in_character and vertical_projection[col] > 0:
            in_character = True
            start_col = col
        elif in_character and vertical_projection[col] == 0:
            in_character = False
            end_col = col
            character_boundaries.append((start_col, end_col))

    # 提取每个字符
    characters = []
    for (start_col, end_col) in character_boundaries:
        character = cropped_plate[:, start_col:end_col]
        characters.append(character)

    return characters

def normalize_character_size(character, target_size=(8, 16)):
    normalized_character = cv2.resize(character, target_size, interpolation=cv2.INTER_AREA)
    return normalized_character
# 使用 Tesseract OCR 辨識字符
def recognize_characters(characters):
     recognized_text = ""
     for char in characters:
         text = pytesseract.image_to_string(char, config='--psm 10 --oem 3')
         recognized_text += text.strip() + " "

     return recognized_text.strip()

def show_video():
    # 视窗名称
    window_name = "daniel car detection"
    # 创建视窗
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # 视窗大小
    cv2.resizeWindow(window_name, 1280, 720)  

    kernel_left = np.array([-1, 0, 1])
    kernel_right = np.array([1, 0, -1])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera. Check your RTSP stream.")
            break
        else:
            frame = cv2.pyrDown(frame)  

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized_frame = cv2.equalizeHist(gray_frame)

            edges_left = cv2.filter2D(equalized_frame, -1, kernel_left)
            edges_right = cv2.filter2D(equalized_frame, -1, kernel_right)
            edges = edges_left + edges_right

            smoothed_edges = cv2.GaussianBlur(edges, (5, 5), 0)
            _, binary_edges = cv2.threshold(smoothed_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(binary_edges, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 2 < aspect_ratio < 5 and w * h > 1000:  
                    license_plate = frame[y:y + h, x:x + w]
                    binary_license_plate = preprocess_license_plate(license_plate)
                    characters = character_segmentation(binary_license_plate)
                    normalized_characters = []
                    for char in characters:
                        normalized_char = normalize_character_size(char)
                        normalized_characters.append(normalized_char)
                    
                    recognized_text = recognize_characters(normalized_characters)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, recognized_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                  

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def index():
    return "Video feed is being shown in a separate window. Close the window to stop the feed."

if __name__ == "__main__":
    video_thread = threading.Thread(target=show_video)
    video_thread.start()
    try:
        app.run(host="0.0.0.0", port=8088, debug=True)
    except Exception as e:
        print(f"Error: {e}")
