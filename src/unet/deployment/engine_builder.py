import tensorrt as trt
from unet.deployment.config import TRTConfig

class UNetEngineBuilder:
    def __init__(self, config: TRTConfig):
        self.config = config
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        
    def build_engine(self, onnx_path: str):
        """Build TensorRT engine from ONNX model"""
        network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
            
        # Configure builder
        config = self.builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size
        
        if self.config.fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
            
        # Build and save engine
        engine = self.builder.build_engine(network, config)
        return engine
