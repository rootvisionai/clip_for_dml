import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from timm.models.swin_transformer import _create_swin_transformer
from collections import OrderedDict


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class DoublePool(nn.Module):
    def __init__(self, out_size=1):
        super().__init__()
        self.out_size = out_size
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.avg_pooling(x)+self.max_pooling(x)
        return x

class Attention(nn.Module):
    def __init__(self, pre_dim=2048, dim=512):
        super().__init__()
        self.pre_dim = pre_dim
        self.dim = dim

        self.to_attentions = nn.Sequential(nn.Linear(pre_dim, int(pre_dim / dim), bias=False),
                                           nn.Softmax(dim=-1))

    def forward(self, x):
        x = l2_norm(x)
        a = self.to_attentions(x)
        x = x.reshape(x.shape[0], int(self.pre_dim / self.dim), self.dim)
        x = x * a[..., None]
        return x.sum(1)


class Base(nn.Module):
    def get_params(self):
        return [
            {"params": self.model.embedding.parameters()},
            {"params": self.model.attention.parameters()}
        ]

    def train(self, train=True):
        if train:
            self.model.eval()
            self.model.embedding.train()
            self.model.attention.train()
        else:
            self.model.train(False)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.model.embedding(x)
        x = self.model.attention(x)
        x = self.classifier(x)
        return x

class TimmSwinTiny(Base):
    def __init__(self, embedding_size, pretrained=True, is_norm=True, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(TimmSwinTiny, self).__init__()
        self.embedding_size = embedding_size
        self.is_norm = is_norm
        self.embedding_size = embedding_size

        model_kwargs = dict(
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            num_classes=0,
            cifar=False,
            zero_init_residual=False
        )
        self.model = _create_swin_transformer("swin_tiny_patch4_window7_224", pretrained=pretrained, **model_kwargs)
        self.num_ftrs = 768
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads, bias=True)
        self._initialize_weights(self.model.embedding)

        self.classifier = nn.Identity()

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=self.embedding_size)
        else:
            self.model.attention = nn.Identity()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self, embedding):
        init.kaiming_normal_(embedding.weight, mode='fan_out')
        try:
            init.constant_(embedding.bias, 0)
        except:
            print("No bias for this layer.")

    def load_ssl_checkpoint(self, path, device="cpu"):

        print("PARAMETERS LOADED FROM SSL CHECKPOINT:")
        checkpoint = torch.load(path, torch.device(device))
        state_dict = checkpoint["state_dict"]

        new_state_dict = OrderedDict()
        model_state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            if "backbone" in k.split(".")[0:2] or "module" in k.split(".")[0:2]:
                k = k.replace("backbone.", "").replace("module.", "")
            if k in model_state_dict:
                new_state_dict[k] = v
                print(f"---> {k}, {v.shape}")
            else:
                print(f"|||| {k}, {v.shape}")
        state_dict = new_state_dict

        self.model.load_state_dict(state_dict, strict=False)

    def get_params(self):
        return [
            {"params": self.model.embedding.parameters()},
            {"params": self.model.attention.parameters()}
        ]

    def train(self, train=True):
        if train:
            self.model.eval()
            self.model.embedding.train()
            self.model.attention.train()
        else:
            self.model.train(False)

    def forward_features(self, x):
        with torch.no_grad():
            x = self.model.patch_embed(x)
            if self.model.absolute_pos_embed is not None:
                x = x + self.model.absolute_pos_embed
            x = self.model.pos_drop(x)
            x = self.model.layers(x)
            x = self.model.norm(x)  # B L C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1) + x.max(dim=1)[0]
        x = self.model.embedding(x)
        x = self.model.attention(x)
        x = self.classifier(x)
        return x

class SwinTransformer(Base):
    def __init__(self, arch, embedding_size, pretrained=True, is_norm=True, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(SwinTransformer, self).__init__()

        arch_options = {
            "tiny" : models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 \
                if pretrained and arch == "tiny" else None),
            "small": models.swin_s(weights=models.swin_transformer.Swin_S_Weights.IMAGENET1K_V1 \
                if pretrained and arch == "small" else None),
            "base" : models.swin_b(weights=models.swin_transformer.Swin_B_Weights.IMAGENET1K_V1 \
                if pretrained and arch == "base" else None)
        }

        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.head.in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads \
                if attention else self.embedding_size)
        self.model.head = torch.nn.Identity()
        self.classifier = nn.Identity()
        self.model.avgpool = DoublePool(1)

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class ConvNext(Base):
    def __init__(self, arch, embedding_size, pretrained=True, is_norm=False, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(ConvNext, self).__init__()

        arch_options = {
            "tiny" : models.convnext_tiny(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 \
                                              if pretrained and arch == "tiny" else None),
            "small": models.convnext_small(models.ConvNeXt_Small_Weights.IMAGENET1K_V1 \
                                               if pretrained and arch == "small" else None),
            "base" : models.convnext_base(models.ConvNeXt_Base_Weights.IMAGENET1K_V1 \
                                              if pretrained and arch == "base" else None),
            "large": models.convnext_large(models.ConvNeXt_Large_Weights.IMAGENET1K_V1 \
                                               if pretrained and arch == "large" else None),
        }

        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.classifier[-1].in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads \
            if attention else self.embedding_size)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.model.classifier[-1] = torch.nn.Identity()
        self.model.avgpool = DoublePool(1)

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class ResNet(Base):
    def __init__(self, arch, embedding_size, pretrained=True, is_norm=False, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(ResNet, self).__init__()

        arch_options = {
            "resnet18" : models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1 \
                                             if pretrained and arch == "resnet18" else None),
            "resnet34": models.resnet34(models.ResNet34_Weights.IMAGENET1K_V1 \
                                            if pretrained and arch == "resnet34" else None),
            "resnet50" : models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2 \
                                             if pretrained and arch == "resnet50" else None),
            "resnet101": models.resnet101(models.ResNet101_Weights.IMAGENET1K_V2 \
                                              if pretrained and arch == "resnet101" else None),
            "resnet152": models.resnet152(models.ResNet152_Weights.IMAGENET1K_V2 \
                                              if pretrained and arch == "resnet152" else None),
        }

        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.fc.in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads \
            if attention else self.embedding_size)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = DoublePool(1)

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class SqueezeNet(Base):
    def __init__(self, arch, embedding_size, pretrained=True, is_norm=False, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(SqueezeNet, self).__init__()

        arch_options = {
            "squeezenet1_0" : models.squeezenet1_0(models.SqueezeNet1_0_Weights.IMAGENET1K_V1 \
                                                       if pretrained and arch == "squeezenet1_0" else None),
            "squeezenet1_1": models.squeezenet1_1(models.SqueezeNet1_0_Weights.IMAGENET1K_V1 \
                                                      if pretrained and arch == "squeezenet1_1" else None)
        }

        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.classifier[-1].in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads \
            if attention else self.embedding_size)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.model.classifier = torch.nn.Identity()

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class EffNetV2(Base):
    def __init__(self, arch, embedding_size, pretrained=True, is_norm=False, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(EffNetV2, self).__init__()

        arch_options = {
            "efficientnet_v2_s": models.efficientnet_v2_s(models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 \
                                                            if pretrained and arch == "efficientnet_v2_s" else None),
            "efficientnet_v2_m": models.efficientnet_v2_s(models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 \
                                                            if pretrained and arch == "efficientnet_v2_m" else None),
            "efficientnet_v2_l": models.efficientnet_v2_s(models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 \
                                                            if pretrained and arch == "efficientnet_v2_l" else None)
        }

        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.classifier[-1].in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads \
            if attention else self.embedding_size)
        self.is_norm = is_norm
        self.embedding_size = embedding_size

        self.model.classifier = torch.nn.Identity()
        self.model.avgpool = DoublePool(1)

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)


class ViT(Base):
    def __init__(self, arch, embedding_size, pretrained=True, is_norm=False, bn_freeze=False,
                 attention=False, attention_heads=1):
        super(ViT, self).__init__()

        arch_options = {
            "vit_b_16": models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 \
                                        if pretrained and arch == "vit_b_16" else None),
            "vit_b_32": models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1 \
                                        if pretrained and arch == "vit_b_32" else None),
            "vit_l_16": models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1 \
                                        if pretrained and arch == "vit_l_16" else None),
            "vit_l_32": models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1 \
                                        if pretrained and arch == "vit_l_32" else None),
            "vit_h_14": models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_V1 \
                                        if pretrained and arch == "vit_h_14" else None)
        }

        self.embedding_size = embedding_size
        self.model = arch_options[arch]
        self.num_ftrs = self.model.heads.head.in_features
        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size * attention_heads \
            if attention else self.embedding_size)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.model.heads = torch.nn.Identity()

        if attention:
            self.model.attention = Attention(pre_dim=embedding_size * attention_heads, dim=embedding_size)
        else:
            self.model.attention = nn.Identity()

        self.classifier = nn.Identity()

        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)