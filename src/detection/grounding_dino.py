
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

image_url = "/home/hari/Downloads/pexels-kolkatarchobiwala-18414576.jpg"
image = Image.open(image_url)
# Check for cats and remote controls
text_labels = [["persons holding a banner"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# Retrieve the first image result
result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [int(x) for x in box.tolist()]
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red")
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
image.show()