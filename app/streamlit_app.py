import streamlit as st
import os
st.title("VisionIQ - Video Intelligence Copilot")
support_video_formats = ["MKV","MP4","AVI"]
video_path = st.text_input(label=f"Enter the video {support_video_formats}: ", value="",)
from utils import Config
from src.detection.yolo_infer import YOLOInfer

def validate_video_path(video_path):
    if video_path.strip()[-3:] in support_video_formats:
        return True
    else:
        return False

def inference_with_yolo(video_path):
    yolo = YOLOInfer(Config.YOLO_MODEL, Config.USE_TENSORRT)
    return yolo

if os.path.exists(video_path):
    is_valid = validate_video_path(video_path)
    if is_valid:
        st.write("File exists")
    else:
        st.write("Please enter a video file that endswith .mp4 or .avi or .mkv")
elif not any(video_path):
    st.write("Please enter a valid video path")
else:
    st.write("File does not exist")
