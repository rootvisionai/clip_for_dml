from transformers import CLIPImageProcessor, CLIPModel
import torch
import torch.nn as nn


class CLIP(nn.Module):

    def __init__(
            self,
            loss_function,
            embedding_size=512,
            optimizer="AdamW",
            lr=0.01,
            device="cuda"
    ):

        super().__init__()

        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self.embedding = nn.Linear(list(self.model.children())[-1].out_features, embedding_size, bias=False)
        self.classifier = nn.Identity()

        self.criterion = loss_function
        self.optimizer = optimizer
        self.lr = lr
        self.opt = getattr(torch.optim, self.optimizer)(params=self.get_params(), lr=self.lr)

    def get_params(self):
        return self.embedding.parameters()

    def features(self, images):
        images = [img for img in images]
        with torch.no_grad():
            inputs = self.image_processor.preprocess(images, return_tensors="pt").to(self.device)
            feature = self.model.get_image_features(**inputs)
        return feature

    def forward(self, x):
        x = self.features(x)
        x = self.embedding(x)
        x = self.classifier(x)
        return x

    def inference(self, x):
        with torch.no_grad():
            x = self.forward(x)
        return x

    def training_step(self, x, l):
        x = self.forward(x)
        x = self.criterion(x, l.to(self.device))

        self.opt.zero_grad()
        x.backward()
        self.opt.step()

        return x.item()

