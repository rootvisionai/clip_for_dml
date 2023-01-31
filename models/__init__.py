from .base import SwinTransformer, ConvNext, ResNet, SqueezeNet, EffNetV2, ViT, TimmSwinTiny
from .resnet import Resnet18, Resnet34, Resnet50, Resnet101


# Create the model
def load(cfg, pretrained = True):
    model_embedding_size = int(cfg.embedding_size)
    
    if cfg.backbone == "swin_tiny":
        model = SwinTransformer(
            "tiny",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif cfg.backbone == "swin_small":
        model = SwinTransformer(
            "small",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif cfg.backbone == "swin_base":
        model = SwinTransformer(
            "base",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif cfg.backbone == "convnext_tiny":
        model = ConvNext(
            "tiny",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif cfg.backbone == "convnext_small":
        model = ConvNext(
            "small",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif cfg.backbone == "convnext_base":
        model = ConvNext(
            "base",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif cfg.backbone == "convnext_large":
        model = ConvNext(
            "large",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif "resnet" in cfg.backbone:
        if "old" in cfg.backbone:
            cfg.backbone = cfg.backbone.replace("_old", "")
            if cfg.backbone == "resnet18":
                model = Resnet18(
                    embedding_size=model_embedding_size,
                    pretrained=pretrained,
                    is_norm=cfg.l2_norm,
                    bn_freeze=cfg.bn_freeze,
                    attention=cfg.attention.status,
                    attention_heads=cfg.attention.heads
                )
            elif cfg.backbone == "resnet34":
                model = Resnet34(
                    embedding_size=model_embedding_size,
                    pretrained=pretrained,
                    is_norm=cfg.l2_norm,
                    bn_freeze=cfg.bn_freeze,
                    attention=cfg.attention.status,
                    attention_heads=cfg.attention.heads
                )
            elif cfg.backbone == "resnet50":
                model = Resnet50(
                    embedding_size=model_embedding_size,
                    pretrained=pretrained,
                    is_norm=cfg.l2_norm,
                    bn_freeze=cfg.bn_freeze,
                    attention=cfg.attention.status,
                    attention_heads=cfg.attention.heads
                )
            elif cfg.backbone == "resnet101":
                model = Resnet101(
                    embedding_size=model_embedding_size,
                    pretrained=pretrained,
                    is_norm=cfg.l2_norm,
                    bn_freeze=cfg.bn_freeze,
                    attention=cfg.attention.status,
                    attention_heads=cfg.attention.heads
                )
            else:
                print(f"{cfg.backbone} is not not what is expected.")
        else:
            model = ResNet(
                cfg.backbone,
                model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )
    elif "effnetv2_s" == cfg.backbone:
        model = EffNetV2(
            "efficientnet_v2_s",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif "effnetv2_m" == cfg.backbone:
        model = EffNetV2(
            "efficientnet_v2_m",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif "effnetv2_l" == cfg.backbone:
        model = EffNetV2(
            "efficientnet_v2_l",
            model_embedding_size,
            pretrained=pretrained,
            bn_freeze=cfg.bn_freeze,
            attention=cfg.attention.status,
            attention_heads=cfg.attention.heads
        )
    elif "vit" in cfg.backbone:
        if "b16" in cfg.backbone:
            model = ViT(
                "vit_b_16",
                model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )
        elif "b32" in cfg.backbone:
            model = ViT(
                "vit_b_32",
                model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )
        elif "l16" in cfg.backbone:
            model = ViT(
                "vit_l_16",
                model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )
        elif "l32" in cfg.backbone:
            model = ViT(
                "vit_l_32",
                model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )
        elif "h14" in cfg.backbone:
            model = ViT(
                "vit_h_14",
                model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )
        else:
            print(f"{cfg.backbone} is not not what is expected.")

    elif "timm_swin_tiny" == cfg.backbone:
        model = TimmSwinTiny(
                embedding_size=model_embedding_size,
                pretrained=pretrained,
                bn_freeze=cfg.bn_freeze,
                attention=cfg.attention.status,
                attention_heads=cfg.attention.heads
            )

    else:
        raise ValueError('Unsupported model depth, \
                         must be one of resnet18, resnet34, resnet50, resnet101, resnet152 \
                         sqznet0, sqznet1 \
                         convnext_tiny, convnext_small, convnext_base, convnext_large \
                         swin_tiny, swin_small, swin_base \
                         effnetv2_s, effnetv2_m, effnetv2_l')
    
    return model

# if __name__ == "__main__":
#     import torch
#     from base import SwinTransformer, ConvNext, ResNet, SqueezeNet, EffNetV2
#     class Namespace:
#         def __init__(self, **kwargs):
#             self.__dict__.update(kwargs)
#
#     models_str = 'resnet18, resnet34, resnet50, resnet101, resnet152, \
#                   convnext_tiny, convnext_small, convnext_base, convnext_large, \
#                   swin_tiny, swin_small, swin_base, \
#                   effnetv2_s, effnetv2_m, effnetv2_l'
#
#     statuses = [False, True]
#
#     for elm in models_str.replace(" ","").split(","):
#         for status in statuses:
#             attention = Namespace(status=status, heads=4 if status else 1)
#             cfg = Namespace(
#                 backbone=elm,
#                 embedding_size=64,
#                 attention=attention,
#                 bn_freeze=False
#             )
#             print("UNIT TEST")
#             print(f"CFG: {cfg.__dict__.items()}")
#             print(f"CFG.ATTENTION: {cfg.attention.__dict__.items()}")
#
#             model = load(cfg, pretrained=False)
#             emb = model(torch.rand((1,3,224,224)))
#             assert emb.shape[0] == 1
#             assert emb.shape[-1] == cfg.embedding_size
#             print("TEST FINISHED.")
#             print("_____________")