import cv2
import numpy as np

def process_image(image_path):
    # 读取图像文件
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 缩放图像大小（保留原始大小）
    img = cv2.pyrDown(img)  # 或者使用 cv2.resize()

    # 将图像转换为灰阶图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   

    # 进行直方图均衡化
    equalized_img = cv2.equalizeHist(gray_img)

    img_blur = cv2.GaussianBlur( equalized_img, (19,19), 0)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(equalized_img,30, 150)  # 这里的参数可以根据需要调整

    return img, gray_img, equalized_img,img_blur,edges

def enhance_plate_features( edges):
    # 定义一维遮罩
    mask1 = np.array([-1, 0, 1])
    mask2 = np.array([1, 0, -1])

    # 对图像应用一维遮罩
    vertical_edges1 = cv2.filter2D(equalized_img, -1, mask1)
    vertical_edges2 = cv2.filter2D(equalized_img, -1, mask2)

    # 合并两次过滤的结果
    enhanced_img = cv2.addWeighted(vertical_edges1, 1, vertical_edges2, 1, 0)

    return enhanced_img

def smooth_and_binarize(enhanced_img, threshold=45):
    # Apply Gaussian blur to smooth the image
    
    # Apply binary thresholding with the given threshold value
    _, binary_img = cv2.threshold(enhanced_img, threshold, 255, cv2.THRESH_BINARY)
    
    return binary_img

def morphological_operations(binary_img):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 应用开操作
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 应用闭操作
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)

    return opened_img, closed_img

def extract_plate(closed_img):
    # 计算 X 轴和 Y 轴的投影
    projection_x = np.sum(closed_img, axis=1)
    projection_y = np.sum(closed_img, axis=0)

    # 设定一个阈值，找到车牌的边界
    x_thresh = np.max(projection_x) * 100
    y_thresh = np.max(projection_y) * 200

    x_min = np.argmax(projection_x > x_thresh)
    x_max = len(projection_x) - np.argmax(projection_x[::-1] > x_thresh)
    y_min = np.argmax(projection_y > y_thresh)
    y_max = len(projection_y) - np.argmax(projection_y[::-1] > y_thresh)

    # 提取车牌区域
    plate_img = closed_img[y_min:y_max, x_min:x_max]

    return plate_img, (x_min, y_min, x_max, y_max)

if __name__ == "__main__":
    image_path = r"D:\Car dection\6_plates\-_jpeg.rf.770fd0d513e6577a89b510782485e4dd.jpg"  # 使用上传的图像路径
    original_img, gray_img,equalized_img, img_blur,edges = process_image(image_path)
    enhanced_img = enhance_plate_features(edges)
    binary_img= smooth_and_binarize(enhanced_img)
    opened_img, closed_img = morphological_operations(binary_img)
    plate_img, plate_coords = extract_plate(closed_img)

    # 显示原始图像
    cv2.imshow('Original Image', original_img)
    # 显示灰阶图像
    cv2.imshow('Gray Image', gray_img)
    # 显示直方图均衡化后的图像
    cv2.imshow('Equalized Image', equalized_img)
    cv2.imshow(' img_blur', img_blur)
    # 显示车牌特征增强后的图像
    cv2.imshow('Enhanced Plate Features', enhanced_img)
    # 显示平滑化后的图像
  
    # 显示二值化后的图像
    cv2.imshow('Binary Image', binary_img)
    # 显示开操作后的图像
    cv2.imshow('Opened Image', opened_img)
    # 显示闭操作后的图像
    cv2.imshow('Closed Image', closed_img)
    # 显示Canny边缘检测后的图像
    cv2.imshow('Edges Image', edges)
    # 显示车牌区域
    cv2.imshow('Extracted Plate', plate_img)

    # 绘制车牌区域在原始图像上
    x_min, y_min, x_max, y_max = plate_coords
    cv2.rectangle(original_img, (x_min*2, y_min*2), (x_max*2, y_max*2), (0, 255, 0), 2)
    cv2.imshow('Original Image with Plate', original_img)

    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
