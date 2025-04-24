import torch
from unet.model import UNet

def export_to_onnx(model: UNet, save_path: str):
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
