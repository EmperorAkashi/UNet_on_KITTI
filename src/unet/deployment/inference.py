import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class UNetTRTInference:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
    def infer(self, input_img: np.ndarray):
        """
        CPU (Host)                    GPU (Device)
        [input_img] ----H2D copy---> [d_input]
                                    ↓
                                    [inference]
                                    ↓
        [output] <----D2H copy----- [d_output]        
        """
        # Allocate memory, input and output buffer
        d_input = cuda.mem_alloc(input_img.nbytes)
        d_output = cuda.mem_alloc(self.output_size)
        
        # Copy input to GPU
        cuda.memcpy_htod(d_input, input_img)
        
        # Run inference
        self.context.execute_v2([d_input, d_output])
        
        # Copy output back to CPU
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        return output
