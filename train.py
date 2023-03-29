from YOLOvision import YOLO

if __name__ == "__main__":
    model = YOLO('downloads/YOLOvision-L.pt')
    print(sum(f.numel() for f in model.model.parameters()) / 1e6)
