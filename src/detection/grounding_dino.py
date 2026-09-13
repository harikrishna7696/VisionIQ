import torch
from PIL import ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDino:
    def __init__(self, model_id: str):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

    def detect(self, image, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(
                "Grounding DINO requires a non-empty text caption; "
                f"received {type(caption).__name__}."
            )

        # The tokenizer accepts a string or a flat list of strings. A nested
        # list such as [[caption]] is only suitable as post-processing label
        # metadata and causes TextEncodeInput errors during tokenization.
        text_prompt = caption.strip()

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # Retrieve the first image result
        result = results[0]
        draw = ImageDraw.Draw(image)
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [int(x) for x in box.tolist()]
            draw.rectangle(box, outline="red")
            print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        return image
