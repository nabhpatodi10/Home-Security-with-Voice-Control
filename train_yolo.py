import torch
from ultralytics import YOLO

def main():
    # Load the YOLO model
    model = YOLO('yolo11x.pt')  # Replace with your YOLO version if necessary
    
    # Fine-tune YOLOv8 on a custom dataset
    output = model.train(data="Weapon_Dataset/data.yaml", epochs=5, imgsz=640)

if __name__ == '__main__':
    # This protects the code from being executed on subprocesses on Windows
    torch.multiprocessing.set_start_method('spawn', force=True)  # Set multiprocessing method to 'spawn'
    main()
