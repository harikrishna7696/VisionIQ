"""
VisionIQ - Video Intelligence Copilot

Streamlit application for running YOLO-based object detection on a video.

Features:
    - Accepts local video file paths.
    - Validates supported video formats.
    - Verifies that the video exists before inference.
    - Runs YOLO inference using the configured model.
    - Displays application status and validation errors in Streamlit.

Supported formats:
    - .mp4
    - .avi
    - .mkv
"""

from pathlib import Path

import streamlit as st

from utils import Config
from vector_db import LanceDB
from src.detection.yolo_infer import YOLOInfer


# ---------------------------------------------------------------------------
# Application Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "VisionIQ - Video Intelligence Copilot"

SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mkv"}

SUPPORTED_VIDEO_FORMATS_TEXT = ", ".join(
    extension.upper().replace(".", "") 
    for extension in sorted(SUPPORTED_VIDEO_FORMATS)
)


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎥",
    layout="wide",
)

st.title(APP_TITLE)
st.caption("YOLO-powered video intelligence and object detection")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def validate_video_path(video_path: str) -> tuple[bool, str]:
    """
    Validate a video path.

    The function checks:
        1. Whether the path is non-empty.
        2. Whether the path points to an existing file.
        3. Whether the file has a supported video extension.

    Args:
        video_path: Path to the video file.

    Returns:
        A tuple containing:
            - bool: True when the path is valid, otherwise False.
            - str: Human-readable validation message.
    """
    path = Path(video_path.strip())

    if not video_path.strip():
        return False, "Please enter a video file path."

    if not path.exists():
        return False, f"Video file does not exist: `{path}`"

    if not path.is_file():
        return False, f"The specified path is not a file: `{path}`"

    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
        return False, f"Unsupported video format. Supported formats: {supported}"

    return True, "Video path is valid."

def is_valid_prompt(prompt_text: str):
    if not prompt_text.strip():
        return False, "Please enter a valid prompt text."
    elif len(prompt_text.strip()) < 4:
        return False, "Please enter a prompt > 4 words."
    return True, "Prompt text is valid."

@st.cache_resource
def load_yolo_model() -> YOLOInfer:
    """
    Initialize and cache the YOLO inference engine.

    Streamlit reruns the application whenever the UI state changes.
    Caching the YOLO model prevents unnecessarily reloading the model
    on every rerun.

    Returns:
        Initialized YOLOInfer instance.
    """



def run_inference(video_path: str, prompt_text: str) -> None:
    """
    Run YOLO inference on the supplied video.

    Args:
        video_path: Valid path to the input video.

    Raises:
        Exception: Any inference-related exception is handled by the
                   Streamlit UI layer.
    """
    try:
        with st.spinner("Running YOLO inference..."):
            yolo = YOLOInfer(
                            Config.YOLO_MODEL,
                            Config.USE_TENSORRT,
                            Config.CONFIDENCE,
                        )
        with st.spinner("Processing video..."):
            yolo.read_video_get_detections(video_path, st, prompt_text)
        st.success("Video inference completed successfully. saved image and text embedding to VectorDB")

        input_prompt = st.text_input(
            label=f"Describe what you are searching from given video",
            help=f"Describe what you are searching from given video. ",
            value="is any person wearing a hat"
        )
        if input_prompt is None:
            st.info("Enter a prompt")
        else:
            img, caption = yolo.fetch_frames_match_to_prompt_text(input_prompt, st)
            st.image(img, caption=caption)
    except Exception as exc:
        st.error("An error occurred while processing the video.")
        st.exception(exc)


# ---------------------------------------------------------------------------
# User Interface
# ---------------------------------------------------------------------------

video_path = st.text_input(
    label=f"Enter video path ({SUPPORTED_VIDEO_FORMATS_TEXT})",
    placeholder="/path/to/video.mp4",
    value="/home/hari/Videos/vlc-record-2026-09-13-16h02m26s-vlc-record-2026-09-11-22h54m38s-6655746-hd_1920_1080_30fps.mp4-.mp4-.mp4",
    help=(
        "Enter the path to a local video file. "
        "Supported formats: MP4, AVI, MKV."
    ),
)
prompt_text = st.text_input(
    label=f"Enter prompt text",
    placeholder="Enter prompt text",
    value="Describe the action, appearance, etc.",
    help=(
    "Ex: Describe the activity of the main person or vehicle "
    "in this image in one short sentence. "
    "Focus on the action and object color, not appearance. "
    "If the activity is uncertain, say so."
    )
)

try:
    if not video_path.strip():
        st.info("Please enter a valid video file path.")

    else:
        is_valid, message = validate_video_path(video_path)
        is_text_valid, tex_message = is_valid_prompt(prompt_text)

        if video_path is None or prompt_text is None:
            st.warning("Please enter a valid video file path.")
        if not is_text_valid:
            st.warning(tex_message)
        elif not is_valid:
            st.warning(message)
        else:
            st.success(message)

            # Normalize whitespace before passing the path to inference.
            normalized_video_path = video_path.strip()

            run_inference(normalized_video_path, prompt_text)
except KeyboardInterrupt as e:
    exit()
