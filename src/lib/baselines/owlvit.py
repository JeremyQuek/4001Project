"""External check: does an open-vocabulary detector (OWL-ViT) find the object in the image?"""
import torch
from PIL import Image


class OwlVitChecker:
    def __init__(self, name="google/owlvit-base-patch32", device=None):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(name)
        self.model = OwlViTForObjectDetection.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def presence(self, image_path, categories):
        """{category: max detection probability over all boxes}."""
        if not categories:
            return {}
        image = Image.open(image_path).convert("RGB")
        queries = [f"a photo of a {c}" for c in categories]
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0]  # [boxes, queries]
        best = torch.sigmoid(logits).max(0).values.cpu().tolist()
        return dict(zip(categories, best))
