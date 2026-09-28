import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
import decord
import numpy as np
from PIL import Image

# 1. Load Model & Processor
bitconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bitconfig,
    device_map="auto"
)

print("ModelLoaded")

processor = AutoProcessor.from_pretrained(model_id)


# ---------------------------------------------------------
# 2. Video
# ---------------------------------------------------------

video_path = "/home/hari/Downloads/18437773-uhd_3840_2160_50fps.mp4"

vr = decord.VideoReader(video_path)

total_frames = len(vr)
fps = vr.get_avg_fps()

duration_secs = total_frames / fps

print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print(f"Duration: {duration_secs:.2f}s")


# ---------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------

CHUNK_DURATION = 1       # seconds
SAMPLE_FPS = 1            # frames per second

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

results = []


# ---------------------------------------------------------
# 4. Process chunks
# ---------------------------------------------------------

for start_sec in range(0, int(duration_secs), CHUNK_DURATION):

    end_sec = min(
        start_sec + CHUNK_DURATION,
        duration_secs
    )

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    step = max(1, int(fps / SAMPLE_FPS))

    frame_indices = list(
        range(start_frame, end_frame, step)
    )

    # -----------------------------------------------------
    # Extract frames
    # -----------------------------------------------------

    frames = vr.get_batch(frame_indices).asnumpy()

    resized_frames = []

    for frame in frames:

        # Decord gives RGB numpy array
        image = Image.fromarray(frame)

        # Resize to 1280x720
        image = image.resize(
            (TARGET_WIDTH, TARGET_HEIGHT),
            Image.Resampling.LANCZOS
        )

        resized_frames.append(image)

    # -----------------------------------------------------
    # Qwen VL input
    # -----------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": resized_frames,
                    "fps": SAMPLE_FPS,
                },
                {
                    "type": "text",
                    "text": (
                        f"This is segment [{start_sec}s - {end_sec}s]. "
                        "Summarize any object entries, departures, "
                        "or critical movements."
                    )
                }
            ]
        }
    ]

    # -----------------------------------------------------
    # Process
    # -----------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128
        )

    output_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    results.append({
        "window": f"{start_sec}s-{end_sec}s",
        "description": output_text
    })

    print(
        f"[{start_sec}s - {end_sec}s]: "
        f"{output_text}"
    )