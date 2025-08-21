"""
Unified adapter utilities for character consistency
Supports both InstantID (for SDXL) and IP-Adapter (for SD 3.5)
"""
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Optional, Union, Dict
import os

# For InstantID
from insightface.app import FaceAnalysis
from diffusers import ControlNetModel

# For IP-Adapter
from transformers import SiglipVisionModel, SiglipImageProcessor


class CharacterConsistencyAdapter:
    """Base class for character consistency adapters"""
    
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
    
    def prepare_images(self, images: List[Image.Image]) -> Dict:
        """Prepare images for the adapter"""
        raise NotImplementedError
    
    def apply_to_pipeline(self, pipe):
        """Apply adapter to the pipeline"""
        raise NotImplementedError


class InstantIDAdapter(CharacterConsistencyAdapter):
    """InstantID adapter for SDXL models"""
    
    def __init__(self, device="cuda", dtype=torch.float16):
        super().__init__(device, dtype)
        self.app = None
        self.controlnet = None
        
    def load_models(self, antelopev2_path="models/antelopev2", instantid_path="models/instantid"):
        """Load InstantID models"""
        print("Loading InstantID models...")
        
        # Load InsightFace model for face detection
        self.app = FaceAnalysis(
            name='antelopev2',
            root=os.path.dirname(antelopev2_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load ControlNet for InstantID
        self.controlnet = ControlNetModel.from_pretrained(
            os.path.join(instantid_path, "ControlNetModel"),
            torch_dtype=self.dtype
        ).to(self.device)
        
        print("InstantID models loaded successfully")
        
    def get_face_embedding(self, image: Image.Image):
        """Extract face embedding from image"""
        # Convert PIL to cv2 format
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self.app.get(image_cv2)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        
        # Use the first detected face
        face = faces[0]
        
        # Get face embedding
        face_embedding = face.embedding
        
        # Get face keypoints for ControlNet
        face_kps = face.kps
        
        return {
            "embedding": face_embedding,
            "keypoints": face_kps,
            "bbox": face.bbox
        }
    
    def prepare_images(self, images: List[Image.Image]) -> Dict:
        """Prepare face embeddings from images"""
        embeddings = []
        keypoints = []
        
        for img in images:
            face_data = self.get_face_embedding(img)
            embeddings.append(face_data["embedding"])
            keypoints.append(face_data["keypoints"])
        
        # Average embeddings for multi-image input
        avg_embedding = np.mean(embeddings, axis=0)
        
        return {
            "face_embedding": torch.from_numpy(avg_embedding).to(self.device, self.dtype),
            "face_keypoints": keypoints[0],  # Use first image's keypoints
            "controlnet": self.controlnet
        }
    
    def apply_to_pipeline(self, pipe, face_data: Dict):
        """Apply InstantID to SDXL pipeline"""
        # Load IP-Adapter if available
        try:
            # Get InstantID IP-Adapter path
            hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            instantid_cache = os.path.join(hf_home, 'hub', 'models--InstantX--InstantID')
            
            if os.path.exists(instantid_cache):
                snapshots_dir = os.path.join(instantid_cache, 'snapshots')
                if os.path.exists(snapshots_dir):
                    snapshot = os.listdir(snapshots_dir)[0]
                    adapter_path = os.path.join(snapshots_dir, snapshot)
                    
                    # Load IP-Adapter
                    if hasattr(pipe, 'load_ip_adapter'):
                        pipe.load_ip_adapter(
                            "InstantX/InstantID", 
                            subfolder="", 
                            weight_name="ip-adapter.bin"
                        )
                        pipe.set_ip_adapter_scale(0.8)
                        print("✓ InstantID IP-Adapter loaded")
                    
        except Exception as e:
            print(f"Could not load InstantID IP-Adapter: {e}")
    
    def create_controlnet_conditioning(self, image: Image.Image) -> Image.Image:
        """Create ControlNet conditioning image from face keypoints"""
        try:
            # Convert to opencv format
            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Get face analysis
            faces = self.app.get(image_cv2)
            
            if len(faces) == 0:
                # Fallback to edge detection
                gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                return Image.fromarray(edges_rgb)
            
            # Create keypoint conditioning image
            face = faces[0]  # Use first face
            kps = face.kps.astype(int)
            
            # Create blank image
            conditioning = np.zeros_like(image_cv2)
            
            # Draw face keypoints
            for kp in kps:
                cv2.circle(conditioning, tuple(kp), 3, (255, 255, 255), -1)
            
            # Draw face bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(conditioning, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 2)
            
            # Convert back to RGB
            conditioning_rgb = cv2.cvtColor(conditioning, cv2.COLOR_BGR2RGB)
            return Image.fromarray(conditioning_rgb)
            
        except Exception as e:
            print(f"Error creating ControlNet conditioning: {e}")
            # Fallback to original image
            return image


class IPAdapterSD35(CharacterConsistencyAdapter):
    """IP-Adapter for SD 3.5 models"""
    
    def __init__(self, device="cuda", dtype=torch.float16):
        super().__init__(device, dtype)
        self.image_encoder = None
        self.image_processor = None
        self.adapter_weights = None
        
    def load_models(self, adapter_path="models/ip-adapter-sd35"):
        """Load IP-Adapter models for SD 3.5"""
        print("Loading SD 3.5 IP-Adapter...")
        
        # Load CLIP image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.image_processor = SiglipImageProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        
        # Load adapter weights - try safetensors first, then bin
        adapter_weight_paths = [
            os.path.join(adapter_path, "ip-adapter.safetensors"),
            os.path.join(adapter_path, "ip-adapter_sd3.5.safetensors"),
            os.path.join(adapter_path, "ip-adapter.bin")
        ]
        
        self.adapter_weights = None
        for weight_path in adapter_weight_paths:
            if os.path.exists(weight_path):
                if weight_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    self.adapter_weights = load_file(weight_path)
                    print(f"SD 3.5 IP-Adapter loaded from safetensors: {weight_path}")
                else:
                    import torch
                    self.adapter_weights = torch.load(weight_path, map_location='cpu')
                    print(f"SD 3.5 IP-Adapter loaded from bin: {weight_path}")
                break
        
        if self.adapter_weights is None:
            print(f"Warning: No IP-Adapter weights found in {adapter_path}")
            print(f"Searched for: {adapter_weight_paths}")
    
    def prepare_images(self, images: List[Image.Image]) -> Dict:
        """Prepare image embeddings for SD 3.5"""
        # Process images with CLIP
        pixel_values = self.image_processor(
            images=images,
            return_tensors="pt"
        ).pixel_values.to(self.device, self.dtype)
        
        # Get image embeddings
        with torch.no_grad():
            image_embeds = self.image_encoder(pixel_values).pooler_output
        
        # Average embeddings if multiple images
        if len(images) > 1:
            image_embeds = image_embeds.mean(dim=0, keepdim=True)
        
        return {
            "image_embeds": image_embeds,
            "adapter_weights": self.adapter_weights
        }
    
    def apply_to_pipeline(self, pipe, image_data: Dict):
        """Apply IP-Adapter to SD 3.5 pipeline"""
        # This would modify the SD 3.5 transformer's attention layers
        # to incorporate the image embeddings
        pass


def create_character_adapter(architecture: str, device="cuda", dtype=torch.float16) -> CharacterConsistencyAdapter:
    """Factory function to create appropriate adapter based on architecture"""
    if architecture == "sdxl":
        adapter = InstantIDAdapter(device, dtype)
        adapter.load_models()
        return adapter
    elif architecture == "sd3":
        adapter = IPAdapterSD35(device, dtype)
        adapter.load_models()
        return adapter
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def detect_faces(image: Image.Image, app: FaceAnalysis) -> List[Dict]:
    """Utility function to detect faces in an image"""
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = app.get(image_cv2)
    
    face_info = []
    for face in faces:
        face_info.append({
            "bbox": face.bbox.tolist(),
            "confidence": face.det_score,
            "embedding": face.embedding,
            "keypoints": face.kps
        })
    
    return face_info