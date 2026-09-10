from transformers import CLIPModel, CLIPProcessor
import torch


class ClipEmbeddings:
    def __init__(self, model):
        self.model = CLIPModel.from_pretrained(model)
        self.processor = CLIPProcessor.from_pretrained(model)

    def image_embeddings(self, image):
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model(**inputs)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        embeddings = image_features.cpu().numpy()

        return embeddings