# Architecture Diagram

Input TIFF + Sequence -> Image Encoder (EfficientNet/ResNet) + Sequence Encoder (ESM-2/BiLSTM) -> Fusion (Concat/Attention) -> Localization Head + Alzheimer's Head
