from ultralytics import YOLO

model_name = "yolo26x.pt"
model = YOLO(model_name)
model.export(format="engine", device=0)