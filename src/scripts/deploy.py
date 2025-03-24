import torch
from pathlib import Path
from unet.model import UNet
from unet.deployment.config import TRTConfig
from unet.deployment.engine_builder import UNetEngineBuilder
from unet.deployment.inference import UNetTRTInference
from scripts.export_onnx import export_to_onnx

class UNetDeployment:
    """High-level API for UNet TensorRT deployment workflow"""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "deployment"):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.onnx_path = self.output_dir / "unet.onnx"
        self.engine_path = self.output_dir / "unet_engine.trt"
        self.config = TRTConfig(engine_path=str(self.engine_path))
    
    def export_model(self):
        """Step 1: Export PyTorch model to ONNX"""
        print("Exporting model to ONNX...")
        model = UNet(in_channels=3, num_classes=2)  # Adjust parameters as needed
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()
        
        export_to_onnx(model, str(self.onnx_path))
        print(f"Model exported to {self.onnx_path}")
    
    def build_engine(self):
        """Step 2: Build TensorRT engine"""
        print("Building TensorRT engine...")
        builder = UNetEngineBuilder(self.config)
        engine = builder.build_engine(str(self.onnx_path))
        
        with open(self.engine_path, 'wb') as f:
            f.write(engine.serialize())
        print(f"Engine saved to {self.engine_path}")
    
    def create_inference_session(self) -> UNetTRTInference:
        """Step 3: Create inference session"""
        return UNetTRTInference(str(self.engine_path))
    
    def deploy(self):
        """Run full deployment workflow"""
        self.export_model()
        self.build_engine()
        return self.create_inference_session()

def main():
    # Example usage
    checkpoint_path = "checkpoints/best.pth"
    deployment = UNetDeployment(checkpoint_path)
    
    # Run full deployment workflow
    inference_session = deployment.deploy()
    
    # Example inference (you would implement your own preprocessing)
    # image = prepare_image(...)
    # result = inference_session.infer(image)
    # processed_result = postprocess_output(result)

if __name__ == "__main__":
    main()