import yaml
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

class Qwen_Model:
    def __init__(self, qwen_model):
        self.qwen_model = qwen_model
        self.qwen, self.processor = self.load_qwen_model()

    def load_qwen_model(self):
        qwen_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16,
                                         bnb_4bit_use_double_quant=True)
        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.qwen_model, quantization_config=qwen_config)
        processor = AutoProcessor.from_pretrained(QWEN_MODEL)
        return qwen, processor

    def caption(self, image):
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

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Put tensors on the same device as the model
        inputs = {
            k: v.to(self.qwen.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.qwen.generate(
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

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return answer.strip()



