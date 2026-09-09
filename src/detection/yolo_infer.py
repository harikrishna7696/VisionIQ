import yaml
from ultralytics import YOLO
import os
import streamlit as st

class YOLOInfer:
    def __init__(self, model_path, use_tensorrt):
        self.model_path = self.export_to_trt(model_path)
        self.use_tensorrt = use_tensorrt
        self.yolo = YOLO(self.model_path)


    def export_to_trt(self,model_path):
        if self.use_tensorrt:
            if os.path.exists(model_path):
                return model_path.replace(".pt", ".engine")
            model = YOLO(model_path)
            st.write("Converting to TRT, please wait... it may take a few minutes...")
            model.export(format="engine", device=0)
            return model_path.replace(".pt", ".engine")
        else:
            return model_path



