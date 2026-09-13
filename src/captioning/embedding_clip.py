
from transformers import CLIPProcessor, CLIPModel
import torch

class ClipEmbeddings:
    def __init__(self, model_id):
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def _flatten(self, features):
        tensor = features.detach().cpu()
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        return tensor.float().tolist()

    def img_embeddings(self, img):
        img_inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(**img_inputs)
            image_features = image_features.pooler_output

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return self._flatten(image_features)

    def text_embeddings(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt")

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features.pooler_output

        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return self._flatten(text_features)
