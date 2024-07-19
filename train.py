import ultralytics
from ultralytics import YOLOWorld, YOLO                      # YOLOv8 套件
import multiprocessing                                       # 多執行續套件

if __name__ == "__main__":
    multiprocessing.freeze_support()                         # 避免多執行續造成主程式重複執行

    model = YOLO('yolov8n.pt')                  # Step 1 : 載入預訓練模型

    results = model.train(                                  # Step 2 : 訓練模型
        data = r'dateset\data.yaml',     # - 指定訓練任務檔
        imgsz = 100,                                # - 輸入影像大小
        epochs = 150,                                # - 訓練世代數
        patience = 50,                              # - 等待世代數，無改善就提前結束訓練
        batch = 1,                                  # - 批次大小
        project = "Yolov8_train_res",            # - 專案名稱
        name = "exp_01"                             # - 續練實驗名稱
    )