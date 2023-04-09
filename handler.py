import torch
import io
import base64
from PIL import Image
import logging
import json

from torchvision import datasets, transforms
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ModelHandler(BaseHandler):
    def __init__(self):
        self.class_labels = None

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

    def initialize(self, context):
        super().initialize(context)

        with open("class_labels.json", 'r') as f:
            self.class_labels = json.loads(f.read())

    def preprocess(self, data):
        image = data[0].get("data") or data[0].get("body")

        if isinstance(image, str):
            image = base64.b64decode(image)

        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))

        data_transform = transforms.Compose([
            transforms.Resize(size=(28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        data = data_transform(image)
        data = data.unsqueeze(dim=0)
        return torch.as_tensor(data, device=self.device)

    def postprocess(self, data):
        class_names = self.class_labels["class_labels"]
        return [class_names[torch.argmax(torch.softmax(data, dim=1), dim=1)]]