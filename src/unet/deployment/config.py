from dataclasses import dataclass

@dataclass
class TRTConfig:
    fp16_mode: bool = True
    int8_mode: bool = False
    max_workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 16
    input_shape: tuple = (3, 224, 224)  # Your UNet input shape
    engine_path: str = "unet_engine.trt"
