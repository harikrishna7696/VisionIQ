import os
import time
from pathlib import Path

import streamlit as st

from utils import Config
from src.detection.yolo_infer import YOLOInfer
from src.captioning.qwen_reasoning import OllamaQwenReasoning


# ============================================================
# App configuration
# ============================================================

APP_TITLE = "VisionIQ"
APP_SUBTITLE = "Video Intelligence Copilot"
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mkv"}

DEFAULT_CHAT_GREETING = (
    "Video loaded successfully. Ask me anything about the events, "
    "people, objects, or actions detected in the video."
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling / animations
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #7c3aed;
        --primary-2: #2563eb;
        --bg: #080b14;
        --card: rgba(17, 24, 39, 0.72);
        --border: rgba(255,255,255,0.10);
        --text: #f8fafc;
        --muted: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(124,58,237,.20), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(37,99,235,.18), transparent 30%),
            linear-gradient(135deg, #070a12 0%, #0b1020 50%, #080b14 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(5, 8, 18, .92);
        border-right: 1px solid var(--border);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 26px;
        margin-bottom: 1.4rem;
        background:
            linear-gradient(135deg, rgba(124,58,237,.22), rgba(37,99,235,.12)),
            rgba(15,23,42,.72);
        box-shadow: 0 20px 70px rgba(0,0,0,.28);
        animation: fadeUp .65s ease-out;
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: rgba(124,58,237,.18);
        filter: blur(50px);
        right: -60px;
        top: -100px;
        animation: floatGlow 5s ease-in-out infinite;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.045em;
        margin: 0;
        position: relative;
    }

    .hero-gradient {
        background: linear-gradient(90deg, #c4b5fd, #93c5fd, #e9d5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #cbd5e1;
        margin-top: .55rem;
        font-size: 1.05rem;
        position: relative;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .38rem .75rem;
        border-radius: 999px;
        background: rgba(34,197,94,.10);
        border: 1px solid rgba(34,197,94,.24);
        color: #86efac;
        font-size: .82rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 15px #22c55e;
        animation: pulse 1.6s infinite;
    }

    .glass-card {
        padding: 1.35rem;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--card);
        backdrop-filter: blur(14px);
        box-shadow: 0 14px 45px rgba(0,0,0,.18);
        animation: fadeUp .55s ease-out;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: .8rem;
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: rgba(15,23,42,.62);
        transition: transform .2s ease, border-color .2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(124,58,237,.42);
    }

    .metric-label {
        color: var(--muted);
        font-size: .78rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: .25rem;
    }

    .chat-empty {
        padding: 3rem 1.5rem;
        text-align: center;
        border: 1px dashed rgba(255,255,255,.13);
        border-radius: 20px;
        background: rgba(15,23,42,.42);
        color: #94a3b8;
        animation: fadeIn .7s ease-out;
    }

    .chat-icon {
        font-size: 2.5rem;
        animation: float 3s ease-in-out infinite;
    }

    .tip {
        color: #94a3b8;
        font-size: .82rem;
        line-height: 1.55;
        margin-top: .8rem;
    }

    .stButton > button {
        width: 100%;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 13px;
        padding: .65rem 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        transition: transform .18s ease, box-shadow .18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(124,58,237,.28);
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: rgba(15,23,42,.72);
        border-color: rgba(255,255,255,.10);
        border-radius: 13px;
    }

    div[data-baseweb="select"] > div {
        background: rgba(15,23,42,.72);
        border-radius: 13px;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.35); opacity: .55; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-7px); }
    }

    @keyframes floatGlow {
        0%, 100% { transform: translate(0,0); }
        50% { transform: translate(-25px,20px); }
    }

    /* Keep the chat composer visually integrated with the dark UI. */
    [data-testid="stChatInput"] {
        padding-bottom: .6rem;
    }

    .small-muted {
        color: #64748b;
        font-size: .75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

def init_state():
    defaults = {
        "messages": [],
        "video_loaded": False,
        "video_path": "",
        "video_summary": "",
        "ollama": None,
        "processing": False,
        "processing_error": None,
        "video_processed_at": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# Helpers
# ============================================================

def validate_video_path(video_path: str):
    if not video_path or not video_path.strip():
        return False, "Please enter a video file path."

    path = Path(video_path.strip())

    if not path.exists():
        return False, f"Video file does not exist: `{path}`"

    if not path.is_file():
        return False, f"The specified path is not a file: `{path}`"

    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
        return False, f"Unsupported format. Supported formats: {supported}"

    return True, "Video path is valid."


def is_valid_prompt(prompt_text: str):
    if not prompt_text or not prompt_text.strip():
        return False, "Please enter a prompt."

    if len(prompt_text.strip()) < 4:
        return False, "Please enter a prompt longer than 4 characters."

    return True, "Prompt is valid."


def reset_chat():
    st.session_state.messages = []


def reset_video():
    st.session_state.video_loaded = False
    st.session_state.video_path = ""
    st.session_state.video_summary = ""
    st.session_state.ollama = None
    st.session_state.processing = False
    st.session_state.processing_error = None
    st.session_state.video_processed_at = None
    st.session_state.messages = []


def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content,
    })
# ============================================================
# Video processing
# ============================================================

def process_video(video_path: str, prompt_text: str):
    save_dir = os.path.join(os.getcwd(), "data", "crops")
    os.makedirs(save_dir, exist_ok=True)

    st.session_state.processing = True
    st.session_state.processing_error = None

    try:
        with st.status("🎬 Preparing VisionIQ...", expanded=True) as status:
            st.write("Loading YOLO model...")
            yolo = YOLOInfer(
                Config.YOLO_MODEL,
                Config.USE_TENSORRT,
                Config.CONFIDENCE,
                temporal=True,
            )

            st.write("Analyzing video frames...")
            data, video_summary = yolo.read_video_get_detections(
                video_path.strip(),
                st,
                prompt_text,
                save_dir,
            )

            st.write("Connecting video intelligence to the chat model...")
            ollama = OllamaQwenReasoning(
                Config.QWEN_REASONING,
                video_summary,
            )

            # Critical: persist expensive objects/results across Streamlit reruns.
            st.session_state.ollama = ollama
            st.session_state.video_summary = video_summary
            st.session_state.video_path = video_path.strip()
            st.session_state.video_loaded = True
            st.session_state.video_processed_at = time.strftime("%H:%M:%S")
            st.session_state.messages = []

            add_message("assistant", DEFAULT_CHAT_GREETING)

            status.update(
                label="✅ Video intelligence ready",
                state="complete",
                expanded=False,
            )

    except Exception as exc:
        st.session_state.processing_error = str(exc)
        st.session_state.video_loaded = False
        st.session_state.ollama = None
        st.error("Video processing failed.")
        st.exception(exc)

    finally:
        st.session_state.processing = False


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            🎥 <span class="hero-gradient">VisionIQ</span>
        </div>
        <div class="hero-subtitle">
            Video Intelligence Copilot · YOLO detection + Qwen reasoning
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            AI video analysis workspace
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Workspace")

    mode = st.radio(
        "Choose a workflow",
        ["💬 Chat with Video", "🔎 Video Detection & Search"],
        index=0,
    )

    st.divider()

    if st.session_state.video_loaded:
        st.success("Video ready")

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Loaded video</div>
                <div class="metric-value">{Path(st.session_state.video_path).name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.video_processed_at:
            st.caption(
                f"Processed at {st.session_state.video_processed_at}"
            )

        if st.button("🧹 Clear video & chat"):
            reset_video()
            st.rerun()

        if st.button("🗑️ Clear chat only"):
            reset_chat()
            st.rerun()

    else:
        st.info("Load a video to activate the copilot.")

    st.divider()

    st.markdown("### ✨ Features")
    st.markdown(
        """
        - 🎯 YOLO object detection
        - 🧠 Qwen reasoning
        - 💬 Persistent conversation
        - 🎞️ Temporal video analysis
        - ⚡ Cached session state
        """
    )

    st.markdown(
        '<div class="small-muted">Supported: MP4 · AVI · MKV</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Video setup
# ============================================================

if not st.session_state.video_loaded:
    st.markdown(
        '<div class="glass-card"><div class="section-title">🎬 Load a video</div>',
        unsafe_allow_html=True,
    )

    video_path = st.text_input(
        "Video path",
        placeholder="/path/to/video.mp4",
        value="/home/hari/Downloads/output.mp4",
        label_visibility="collapsed",
    )

    prompt_text = st.text_area(
        "Analysis prompt",
        value="Describe the action, appearance, objects, and important events in the video.",
        height=100,
        help="This prompt guides the initial video analysis.",
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("🚀 Analyze Video", type="primary"):
            valid_path, path_message = validate_video_path(video_path)
            valid_prompt, prompt_message = is_valid_prompt(prompt_text)

            if not valid_path:
                st.error(path_message)
            elif not valid_prompt:
                st.error(prompt_message)
            else:
                process_video(video_path, prompt_text)

    with col2:
        st.markdown(
            """
            <div class="tip">
                <b>Tip</b><br>
                Video analysis runs once. After processing,
                asking new questions will not restart YOLO.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ============================================================
# Main workspace
# ============================================================

left, right = st.columns([2.2, 1], gap="large")

with left:
    if mode == "💬 Chat with Video":
        st.markdown(
            '<div class="glass-card"><div class="section-title">💬 Video Intelligence Chat</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:
            st.markdown(
                """
                <div class="chat-empty">
                    <div class="chat-icon">🧠</div>
                    <h3>Ask your video anything</h3>
                    <p>
                        Try questions about people, actions, objects,
                        sequence of events, or unusual activity.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Persistent history: rendered on every Streamlit rerun.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.markdown("</div>", unsafe_allow_html=True)

        # IMPORTANT:
        # chat_input causes a rerun, but the expensive model is already
        # stored in st.session_state. Therefore the video is NOT reprocessed.
        if prompt := st.chat_input("Ask something about this video..."):
            add_message("user", prompt)

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("🧠 Thinking..."):
                        response = st.session_state.ollama.get_answer(prompt)

                    # Make sure history contains a plain string.
                    if response is None:
                        response = "I couldn't generate a response."

                    response = str(response)
                    st.markdown(response)
                    add_message("assistant", response)

                except Exception as exc:
                    error_message = f"Sorry, I couldn't answer that: `{exc}`"
                    st.error(error_message)
                    add_message("assistant", error_message)

    else:
        st.markdown(
            '<div class="glass-card"><div class="section-title">🔎 Video Detection & Search</div>',
            unsafe_allow_html=True,
        )

        search_prompt = st.text_input(
            "Search the analyzed video",
            placeholder="e.g. is any person wearing a hat?",
        )

        if st.button("🔍 Search Video"):
            if not search_prompt.strip():
                st.warning("Enter a search prompt.")
            else:
                try:
                    yolo = YOLOInfer(
                        Config.YOLO_MODEL,
                        Config.USE_TENSORRT,
                        Config.CONFIDENCE,
                        temporal=True,
                    )

                    with st.spinner("Searching matching frames..."):
                        img, caption = yolo.fetch_frames_match_to_prompt_text(
                            search_prompt,
                            st,
                        )

                    if img is not None:
                        st.image(img, caption=caption, use_container_width=True)
                    else:
                        st.info("No matching frame was returned.")

                except Exception as exc:
                    st.error("Search failed.")
                    st.exception(exc)

        st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">📊 Session</div>
        """,
        unsafe_allow_html=True,
    )

    status = "🟢 Ready" if st.session_state.video_loaded else "⚪ Waiting"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value">{status}</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">Messages</div>
            <div class="metric-value">{len(st.session_state.messages)}</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">Qwen Reasoning</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.video_summary:
        with st.expander("📝 Video summary"):
            st.write(st.session_state.video_summary)

    st.markdown("</div>", unsafe_allow_html=True)
