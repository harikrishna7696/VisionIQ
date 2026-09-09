import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import yaml

def load_models(yolo_model, qwen_model):
    yolo = YOLO(yolo_model)
    qwen_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_use_double_quant=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model, quantization_config=qwen_config)
    processor = AutoProcessor.from_pretrained(QWEN_MODEL)
    return yolo, qwen, processor

def describe_activity(image):
    """
    Ask Qwen to describe the activity in an image.
    """

    prompt = (
        "Describe the activity of the main person or vehicle "
        "in this image in one short sentence. "
        "Focus on the action and object color, not appearance. "
        "If the activity is uncertain, say so."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Put tensors on the same device as the model
    inputs = {
        k: v.to(qwen.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = qwen.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Remove input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs["input_ids"],
            generated_ids
        )
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return answer.strip()


def read_video_get_caption():
    # --------------------------------------------------
    # Video
    # --------------------------------------------------

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # frame_count = int(cv2.CAP_PROP_FRAME_COUNT)
    frame_id = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_id += 1

        # --------------------------------------------------
        # YOLO tracking
        # --------------------------------------------------

        results = yolo.track(frame, persist=True, verbose=False)

        result = results[0]

        if result.boxes is None:
            continue

        boxes = result.boxes

        # --------------------------------------------------
        # Process detections
        # --------------------------------------------------

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i])
            if confidence < CONFIDENCE:
                continue

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            # --------------------------------------------------
            # Only run VLM periodically
            # --------------------------------------------------

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # OpenCV BGR -> RGB
            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(crop_rgb)

            # --------------------------------------------------
            # Qwen
            # --------------------------------------------------

            answer = describe_activity(image)

            track_id = None

            if boxes.id is not None:
                track_id = int(boxes.id[i])

            print(
                f"Frame={frame_id} "
                f"Track={track_id} "
                f"Confidence={confidence:.2f} "
                f"Activity={answer}"
            )


    video.release()


def main():
    read_video_get_caption()

if __name__ == "__main__":
    config = yaml.load(open("config/config.yaml"), Loader=yaml.FullLoader)
    MAX_NEW_TOKENS = config["VLM"]["max_new_tokens"]
    YOLO_MODEL = config["YOLO"]["detection_model"]
    VIDEO_PATH = config["video_path"]
    CONFIDENCE = config["YOLO"]["confidence"]
    USE_TENSORRT = True
    if USE_TENSORRT:
        YOLO_MODEL = YOLO_MODEL.replace(".pt", ".engine")
    QWEN_MODEL = config["VLM"]["model"]

    yolo, qwen, processor = load_models(YOLO_MODEL,QWEN_MODEL)
    main()