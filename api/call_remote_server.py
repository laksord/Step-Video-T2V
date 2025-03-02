import torch
import os
from flask import Flask, Response, jsonify, request, Blueprint
from flask_restful import Api, Resource
import pickle
import argparse
import threading
import torch.distributed as dist
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Initialize distributed environment
def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize distributed environment for multi-GPU processing"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_model_parallel_group():
    """Get or create the model parallel process group."""
    if not hasattr(get_model_parallel_group, "group"):
        get_model_parallel_group.group = dist.new_group()
    return get_model_parallel_group.group

def parsed_args():
    parser = argparse.ArgumentParser(description="Distributed StepVideo API Functions")
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--clip_dir', type=str, default='hunyuan_clip')
    parser.add_argument('--llm_dir', type=str, default='step_llm')
    parser.add_argument('--vae_dir', type=str, default='vae')
    parser.add_argument('--port', type=str, default='8080')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--vae_shards', type=int, default=2, 
                        help="Number of shards to split the VAE model into")
    parser.add_argument('--caption_shards', type=int, default=2, 
                        help="Number of shards to split the caption model into")
    args = parser.parse_args()
    return args


class TensorParallelVAE:
    """Tensor parallel implementation of VAE for multi-GPU usage"""
    def __init__(self, vae_dir, rank: int, world_size: int, version=2):
        self.rank = rank
        self.world_size = world_size
        self.model_parallel_group = get_model_parallel_group()
        self.vae = self.build_vae(vae_dir, version)
        self.scale_factor = 1.0

    def build_vae(self, vae_dir, version=2):
        from stepvideo.vae.vae import AutoencoderKL
        (model_name, z_channels) = ("vae_v2.safetensors", 64) if version == 2 else ("vae.safetensors", 16)
        model_path = os.path.join(vae_dir, model_name)
        
        # Load model on current device
        device = torch.device(f"cuda:{self.rank}")
        dtype = torch.bfloat16
        
        # Create model with support for distributed execution
        model = AutoencoderKL(
            z_channels=z_channels,
            model_path=model_path,
            version=version,
            world_size=self.world_size  # Pass world_size to enable distributed features
        ).to(dtype).to(device).eval()
        
        print(f"Rank {self.rank}: Initialized VAE")
        return model
 
    def decode(self, samples, *args, **kwargs):
        with torch.no_grad():
            try:
                dtype = next(self.vae.parameters()).dtype
                device = next(self.vae.parameters()).device
                
                # Here we ensure the samples are on the right device for this rank
                samples = samples.to(dtype).to(device) / self.scale_factor
                
                # Use the decoder with distributed processing
                # The decoder internally handles tensor splitting and cross-GPU communication
                samples = self.vae.decode(samples)
                
                if hasattr(samples, 'sample'):
                    samples = samples.sample
                
                # Gather results from all ranks
                if self.world_size > 1:
                    # Create a list to store outputs from all ranks
                    gathered_samples = [torch.zeros_like(samples) for _ in range(self.world_size)]
                    dist.all_gather(gathered_samples, samples, group=self.model_parallel_group)
                    
                    # Process only on rank 0 to avoid duplicate work
                    if self.rank == 0:
                        # Combine results (implementation depends on how the model shards data)
                        # This is a simplified version - actual implementation may differ
                        samples = torch.cat(gathered_samples, dim=0)
                
                return samples
            except Exception as e:
                print(f"Rank {self.rank} caught exception in VAE decode: {e}")
                torch.cuda.empty_cache()
                return None


class DistributedVAEWrapper:
    """Wrapper to manage distributed VAE execution"""
    def __init__(self, vae_pipeline, world_size):
        self.vae_pipeline = vae_pipeline
        self.world_size = world_size
        self.lock = threading.Lock()
        
    def decode(self, samples, *args, **kwargs):
        with self.lock:
            return self.vae_pipeline.decode(samples, *args, **kwargs)


class DistributedCaptionPipeline:
    """Distributed implementation of Caption Pipeline for multi-GPU usage"""
    def __init__(self, llm_dir, clip_dir, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model_parallel_group = get_model_parallel_group()
        
        # Load models specific to this rank's responsibility
        self.text_encoder = self.build_llm(llm_dir)
        self.clip = self.build_clip(clip_dir)
        
    def build_llm(self, model_dir):
        from stepvideo.text_encoder.stepllm import STEP1TextEncoder
        device = torch.device(f"cuda:{self.rank}")
        dtype = torch.bfloat16
        
        # Initialize the text encoder on this device
        text_encoder = STEP1TextEncoder(model_dir, max_length=320).to(dtype).to(device).eval()
        print(f"Rank {self.rank}: Initialized text encoder")
        return text_encoder
        
    def build_clip(self, model_dir):
        from stepvideo.text_encoder.clip import HunyuanClip
        device = torch.device(f"cuda:{self.rank}")
        
        # Initialize the CLIP encoder on this device
        clip = HunyuanClip(model_dir, max_length=77).to(device).eval()
        print(f"Rank {self.rank}: Initialized CLIP encoder")
        return clip
 
    def embedding(self, prompts, *args, **kwargs):
        with torch.no_grad():
            try:
                # Process based on rank to distribute workload
                batch_size = len(prompts) if isinstance(prompts, list) else 1
                
                # Simple strategy: split prompts among ranks
                if isinstance(prompts, list) and self.world_size > 1:
                    # Determine this rank's portion of the prompts
                    items_per_rank = (batch_size + self.world_size - 1) // self.world_size
                    start_idx = self.rank * items_per_rank
                    end_idx = min(start_idx + items_per_rank, batch_size)
                    
                    if start_idx < batch_size:
                        # This rank has work to do
                        rank_prompts = prompts[start_idx:end_idx]
                        
                        # Process this rank's prompts
                        y, y_mask = self.text_encoder(rank_prompts)
                        clip_embedding, _ = self.clip(rank_prompts)
                        
                        # Prepare for all_gather
                        len_clip = clip_embedding.shape[1]
                        y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)
                        
                        # Prepare data dictionary for this rank
                        local_data = {
                            'y': y.detach().cpu(),
                            'y_mask': y_mask.detach().cpu(),
                            'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
                        }
                    else:
                        # This rank has no work (can happen with uneven distribution)
                        local_data = None
                    
                    # Gather results from all ranks
                    gathered_data = [None] * self.world_size
                    dist.all_gather_object(gathered_data, local_data, group=self.model_parallel_group)
                    
                    # Combine results (on all ranks to avoid additional communication)
                    if self.rank == 0:
                        valid_data = [d for d in gathered_data if d is not None]
                        
                        # Combine tensors from different ranks
                        combined_y = torch.cat([d['y'] for d in valid_data], dim=1)
                        combined_y_mask = torch.cat([d['y_mask'] for d in valid_data], dim=0)
                        combined_clip = torch.cat([d['clip_embedding'] for d in valid_data], dim=0)
                        
                        data = {
                            'y': combined_y,
                            'y_mask': combined_y_mask,
                            'clip_embedding': combined_clip
                        }
                        return data
                    else:
                        # Non-zero ranks return None, main rank will handle result
                        return None
                else:
                    # For single prompt or when world_size=1, process normally
                    y, y_mask = self.text_encoder(prompts)
                    clip_embedding, _ = self.clip(prompts)
                    
                    len_clip = clip_embedding.shape[1]
                    y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)
                    
                    data = {
                        'y': y.detach().cpu(),
                        'y_mask': y_mask.detach().cpu(),
                        'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
                    }
                    return data
                
            except Exception as err:
                print(f"Rank {self.rank} caught exception in caption embedding: {err}")
                return None


class DistributedCaptionWrapper:
    """Wrapper to manage distributed caption execution"""
    def __init__(self, caption_pipeline, world_size):
        self.caption_pipeline = caption_pipeline
        self.world_size = world_size
        self.lock = threading.Lock()
        
    def embedding(self, prompts, *args, **kwargs):
        with self.lock:
            return self.caption_pipeline.embedding(prompts, *args, **kwargs)


# API Resource handlers
class VAEapi(Resource):
    def __init__(self, vae_wrapper):
        self.vae_wrapper = vae_wrapper
        
    def get(self):
        try:
            feature = pickle.loads(request.get_data())
            feature['api'] = 'vae'
        
            feature = {k:v for k, v in feature.items() if v is not None}
            video_latents = self.vae_wrapper.decode(**feature)
            response = pickle.dumps(video_latents)

        except Exception as e:
            print(f"Caught Exception in VAE API: {e}")
            return Response(str(e).encode())
        
        return Response(response)


class Captionapi(Resource):
    def __init__(self, caption_wrapper):
        self.caption_wrapper = caption_wrapper
        
    def get(self):
        try:
            feature = pickle.loads(request.get_data())
            feature['api'] = 'caption'
        
            feature = {k:v for k, v in feature.items() if v is not None}
            embeddings = self.caption_wrapper.embedding(**feature)
            response = pickle.dumps(embeddings)

        except Exception as e:
            print(f"Caught Exception in Caption API: {e}")
            return Response(str(e).encode())
        
        return Response(response)


class DistributedRemoteServer:
    def __init__(self, args) -> None:
        # Initialize distributed processing
        init_distributed(args.rank, args.world_size)
        
        self.rank = args.rank
        self.world_size = args.world_size
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)
        
        # Initialize models with tensor parallelism
        if args.rank == 0:
            # Only create the VAE pipeline on rank 0
            self.vae_pipeline = TensorParallelVAE(
                vae_dir=os.path.join(args.model_dir, args.vae_dir),
                rank=args.rank,
                world_size=args.world_size
            )
            vae_wrapper = DistributedVAEWrapper(self.vae_pipeline, args.world_size)
            api.add_resource(
                VAEapi,
                "/vae-api",
                resource_class_args=[vae_wrapper],
            )

            # Only create the caption pipeline on rank 0
            self.caption_pipeline = DistributedCaptionPipeline(
                llm_dir=os.path.join(args.model_dir, args.llm_dir), 
                clip_dir=os.path.join(args.model_dir, args.clip_dir),
                rank=args.rank,
                world_size=args.world_size
            )
            caption_wrapper = DistributedCaptionWrapper(self.caption_pipeline, args.world_size)
            api.add_resource(
                Captionapi,
                "/caption-api",
                resource_class_args=[caption_wrapper],
            )
            
            print(f"Rank {args.rank}: API endpoints initialized")

    def run(self, host="0.0.0.0", port=8080):
        if self.rank == 0:
            # Only the main rank serves the API
            self.app.run(host, port=port, threaded=True, debug=False)
        else:
            # Other ranks wait for work through distributed communication
            while True:
                dist.barrier()  # Wait for communication from rank 0


if __name__ == "__main__":
    args = parsed_args()
    server = DistributedRemoteServer(args)
    server.run(host="0.0.0.0", port=int(args.port))
