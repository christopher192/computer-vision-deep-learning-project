from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")

    # Train the model
    model.train(data = "coco8-seg.yaml", epochs = 5, imgsz = 640)

if __name__ == "__main__":
    main()