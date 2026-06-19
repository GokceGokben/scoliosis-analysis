from ultralytics import YOLO

def start_training():
    # 1.Upload the model 
    model = YOLO('yolov8n.pt') 

    # 2. Start the training only with the necessary parameters
    model.train(
        data='src/dataset.yaml',
        epochs=50,
        imgsz=512,
        batch=16,
        device='cpu'  # write 'cpu' if no NVIDIA GPU is available
    )

if __name__ == "__main__":
    start_training()