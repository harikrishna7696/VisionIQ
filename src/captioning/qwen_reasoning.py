from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import gc
import decord

class QwenReasoning:
    def __init__(self, model_id,context):
        qwen_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16,
                                         bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto",
                                                          device="auto",
                                                          quantization_config=qwen_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.context = context

    def ask_vide(self, question):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a video analysis assistant. "
                    "Answer questions using only the provided video analysis. "
                    "If the information is not available, say that it cannot "
                    "be determined from the available video analysis."
                )
            },
            {
                "role": "user",
                "content": f"""
        Video analysis:

        {self.context}

        Question:
        {question}
        """
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return answer.strip()

class OllamaQwenReasoning:
    def __init__(self, model_id,summary):
        self.model = ChatOllama(model=model_id,
                                temperature=1.0)
        self.video_summary = summary
        self.messages = ChatPromptTemplate.from_messages([
            ("system",
                    """You are a video analysis assistant. 
                    Answer questions using only the provided video analysis. 
                    If the information is not available, say that it cannot 
                    be determined from the available video analysis.
                    
                    video_summary:
                    {video_summary}
                    question:
                    {question}
                    """
             )
        ])

    def get_answer(self, question):
        print("Video summary:")
        print(self.video_summary)
        prompt = self.messages.invoke({
            "video_summary": self.video_summary,
            "question": question
        })

        answer = self.model.invoke(prompt)

        return answer.content.strip()