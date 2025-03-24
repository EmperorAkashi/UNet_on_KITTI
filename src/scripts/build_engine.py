from unet.deployment.config import TRTConfig
from unet.deployment.engine_builder import UNetEngineBuilder

def main():
    config = TRTConfig()
    builder = UNetEngineBuilder(config)
    
    # Build engine
    engine = builder.build_engine("unet.onnx")
    
    # Save engine
    with open(config.engine_path, 'wb') as f:
        f.write(engine.serialize())
