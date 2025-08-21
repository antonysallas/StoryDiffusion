"""
TensorRT Engine Builder for Stable Diffusion 3.5
Handles TensorRT engine building and caching for optimal performance
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

logger = logging.getLogger(__name__)

class TensorRTEngineBuilder:
    """
    Builds and manages TensorRT engines for SD 3.5 ONNX models
    """
    
    def __init__(self, cache_dir: str = "./tensorrt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Please install tensorrt package.")
        
        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
    def _get_engine_path(self, onnx_path: str, precision: str, batch_size: int = 1) -> Path:
        """Generate engine file path based on ONNX model and settings"""
        onnx_path = Path(onnx_path)
        model_hash = hashlib.md5(str(onnx_path).encode()).hexdigest()[:8]
        engine_name = f"{onnx_path.stem}_{precision}_bs{batch_size}_{model_hash}.trt"
        return self.cache_dir / engine_name
    
    def _get_onnx_metadata(self, onnx_path: str) -> Dict:
        """Extract metadata from ONNX model for cache validation"""
        import onnx
        model = onnx.load(onnx_path)
        
        metadata = {
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "domain": model.domain,
            "model_version": model.model_version,
            "doc_string": model.doc_string,
            "graph_name": model.graph.name,
            "input_names": [input.name for input in model.graph.input],
            "output_names": [output.name for output in model.graph.output],
        }
        return metadata
    
    def build_engine(
        self, 
        onnx_path: str, 
        precision: str = "fp16",
        batch_size: int = 1,
        workspace_size: int = 8 * 1024**3,  # 8GB
        optimization_level: int = 5,
        force_rebuild: bool = False
    ) -> str:
        """
        Build TensorRT engine from ONNX model
        
        Args:
            onnx_path: Path to ONNX model
            precision: "fp16", "fp32", or "int8"
            batch_size: Static batch size for optimization
            workspace_size: Maximum workspace size in bytes
            optimization_level: TensorRT optimization level (0-5)
            force_rebuild: Force rebuild even if cached engine exists
            
        Returns:
            Path to built TensorRT engine
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        engine_path = self._get_engine_path(onnx_path, precision, batch_size)
        metadata_path = engine_path.with_suffix('.json')
        
        # Check if cached engine exists and is valid
        if not force_rebuild and engine_path.exists() and metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    cached_metadata = json.load(f)
                current_metadata = self._get_onnx_metadata(onnx_path)
                
                if cached_metadata.get("onnx_metadata") == current_metadata:
                    logger.info(f"Using cached TensorRT engine: {engine_path}")
                    return str(engine_path)
            except Exception as e:
                logger.warning(f"Failed to validate cached engine: {e}")
        
        logger.info(f"Building TensorRT engine for {onnx_path.name} with {precision} precision...")
        
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        
        # Set workspace size
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        
        # Set precision
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 calibration would be needed here for full INT8 support
        
        # Set optimization level
        config.builder_optimization_level = optimization_level
        
        # Create network from ONNX
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.trt_logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                error_msgs = []
                for error_idx in range(parser.num_errors):
                    error_msgs.append(str(parser.get_error(error_idx)))
                raise RuntimeError(f"Failed to parse ONNX model: {'; '.join(error_msgs)}")
        
        # Set input shapes for optimization
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = input_tensor.shape
            
            # Create optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Set batch size
            if input_shape[0] == -1:  # Dynamic batch
                min_shape = [1] + list(input_shape[1:])
                opt_shape = [batch_size] + list(input_shape[1:])
                max_shape = [batch_size * 2] + list(input_shape[1:])
            else:
                min_shape = opt_shape = max_shape = list(input_shape)
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        
        # Build engine
        logger.info(f"Building engine... This may take several minutes.")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        with open(engine_path, 'wb') as engine_file:
            engine_file.write(serialized_engine)
        
        # Save metadata
        metadata = {
            "onnx_path": str(onnx_path),
            "precision": precision,
            "batch_size": batch_size,
            "workspace_size": workspace_size,
            "optimization_level": optimization_level,
            "onnx_metadata": self._get_onnx_metadata(onnx_path),
            "tensorrt_version": trt.__version__
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"TensorRT engine built successfully: {engine_path}")
        return str(engine_path)
    
    def load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, 'rb') as engine_file:
            engine_data = engine_file.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        
        return engine
    
    def get_engine_info(self, engine_path: str) -> Dict:
        """Get information about a TensorRT engine"""
        metadata_path = Path(engine_path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

class TensorRTInferenceSession:
    """
    TensorRT inference session wrapper
    """
    
    def __init__(self, engine_path: str):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        
        self.builder = TensorRTEngineBuilder()
        self.engine = self.builder.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Calculate size
            size = trt.volume(tensor_shape) * self.engine.max_batch_size
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, tensor_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def run(self, input_dict: Dict[str, any]) -> Dict[str, any]:
        """Run inference"""
        # Copy input data to GPU
        for i, (name, data) in enumerate(input_dict.items()):
            cuda.memcpy_htod_async(
                self.inputs[i]['device'], 
                data.ravel(), 
                self.stream
            )
        
        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy outputs back
        outputs = {}
        for i, output in enumerate(self.outputs):
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            outputs[f'output_{i}'] = output['host'].copy()
        
        self.stream.synchronize()
        return outputs