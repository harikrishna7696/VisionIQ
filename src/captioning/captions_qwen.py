import yaml
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import decord, gc
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

class Qwen_Model:
    def __init__(self, qwen_model,max_tokens, temporal=False):
        self.qwen_model = qwen_model
        self.temporal = temporal
        self.qwen, self.processor = self.load_qwen_model()
        self.max_tokens = max_tokens

    def load_qwen_model(self):
        qwen_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16,
                                         bnb_4bit_use_double_quant=True)
        if self.temporal:
            qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.qwen_model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.qwen_model, quantization_config=qwen_config)
        processor = AutoProcessor.from_pretrained(self.qwen_model)
        return qwen, processor

    def caption(self, image, prompt: tuple[str]):
        """
            Ask Qwen to describe the activity in an image.
            """
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
                max_new_tokens=self.max_tokens,
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

    def caption_video(self, video_path):
        vr = decord.VideoReader(video_path)
        video_fps = vr.get_avg_fps()
        total_duration_sec = int(len(vr) / video_fps)

        CHUNK_DURATION = 2  # 30 seconds per window
        SAMPLE_FPS = 2  # 2 frames per second (60 frames per chunk)

        video_summary = []

        # 2. Iterate through 30-second chunks
        for start_sec in range(0, total_duration_sec, CHUNK_DURATION):
            end_sec = min(start_sec + CHUNK_DURATION, total_duration_sec)

            # Calculate exact frame indices for this 30s segment at 2 FPS
            start_frame = int(start_sec * video_fps)
            end_frame = int(end_sec * video_fps)
            step = int(video_fps / SAMPLE_FPS)
            frame_indices = list(range(start_frame, end_frame, step))

            # Extract frames as numpy/PIL array
            sampled_frames = vr.get_batch(frame_indices).asnumpy()
            frame_list = [Image.fromarray(frame) for frame in sampled_frames]

            # 3. Format message passing the sampled 30s chunk directly
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frame_list,
                            "fps": SAMPLE_FPS
                        },
                        {
                            "type": "text",
                            "text": f"""
                        This video clip spans {start_sec}s to {end_sec}s.

                        You are an intelligent traffic system. The provided video is a processed clip where each object is overlaid with an ID. You must monitor and take note of all traffic related events. Start each event description with a start and end time stamp of the event, and use object IDs in the event description.
                        """
                        }
                    ]
                }
            ]

            # 4. Tokenize and execute inference
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = self.qwen.generate(**inputs, max_new_tokens=200)
                output = self.processor.batch_decode(
                    generated_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]

            print(f"\n--- [Segment: {start_sec}s - {end_sec}s] ---")
            print(output)
            video_summary.append(output)

        del self.qwen, self.processor
        gc.collect()
        torch.cuda.empty_cache()
        return video_summary



