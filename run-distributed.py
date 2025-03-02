#!/usr/bin/env python3
"""
This script runs the StepVideo text-to-video model with distributed processing
for both the transformer model and the VAE/caption components.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.parallel import initialize_parall_group
from stepvideo.utils import setup_seed


def launch_api_server(rank, world_size, args):
    """
    Launch the API server for VAE and captioning on a specific rank
    """
    # Set environment variables for distributed setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Adjust port for rank to avoid conflicts
    api_port = int(args.api_port) + rank
    
    # Build command to start the API server
    cmd = [
        "python", "-m", "api.call_remote_server",
        f"--model_dir={args.model_dir}",
        f"--clip_dir={args.clip_dir}",
        f"--llm_dir={args.llm_dir}",
        f"--vae_dir={args.vae_dir}",
        f"--port={api_port}",
        f"--rank={rank}",
        f"--world_size={args.api_world_size}"
    ]
    
    # Start the process
    import subprocess
    process = subprocess.Popen(cmd)
    return process, api_port


def run_distributed_pipeline(rank, world_size, args):
    """
    Run the main text-to-video pipeline in a distributed setting
    """
    # Initialize distributed group for transformer processing
    initialize_parall_group(
        ring_degree=args.ring_degree,
        ulysses_degree=args.ulysses_degree,
        tensor_parallel_degree=args.tensor_parallel_degree
    )
    
    # Set device based on local rank
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    
    # Set random seed for reproducibility
    setup_seed(args.seed)
    
    # Create the pipeline
    pipeline = StepVideoPipeline.from_pretrained(args.model_dir).to(dtype=torch.bfloat16, device="cpu")
    
    # Apply tensor parallelism to the transformer if needed
    if args.tensor_parallel_degree > 1:
        from xfuser.model_executor.models.customized.step_video_t2v.tp_applicator import TensorParallelApplicator
        from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
        
        tp_applicator = TensorParallelApplicator(
            get_tensor_model_parallel_world_size(), 
            get_tensor_model_parallel_rank()
        )
        tp_applicator.apply_to_model(pipeline.transformer)
    
    # Move model to device
    pipeline.transformer = pipeline.transformer.to(device)
    
    # Only on rank 0: Launch the distributed API server processes
    api_processes = []
    if rank == 0:
        print(f"Launching {args.api_world_size} API server processes...")
        for api_rank in range(args.api_world_size):
            process, port = launch_api_server(api_rank, args.api_world_size, args)
            api_processes.append((process, port))
        
        # Use port from the first process (rank 0) for API access
        vae_port = api_processes[0][1]
        
        # Setup API URLs
        vae_url = f"{args.api_host}:{vae_port}"
        caption_url = vae_url  # Use same server for both endpoints
        
        print(f"API server running at {vae_url}")
    else:
        # For non-zero ranks, use predefined ports
        base_port = int(args.api_port)
        vae_url = f"{args.api_host}:{base_port}"
        caption_url = vae_url
    
    # Ensure all processes have a consistent view before proceeding
    dist.barrier()
    
    # Setup API connections
    pipeline.setup_api(
        vae_url=vae_url,
        caption_url=caption_url,
    )
    
    # Run inference
    prompt = args.prompt
    videos = pipeline(
        prompt=prompt, 
        num_frames=args.num_frames, 
        height=args.height, 
        width=args.width,
        num_inference_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=prompt[:50]
    )
    
    # Terminate API processes when done (rank 0 only)
    if rank == 0:
        for process, _ in api_processes:
            process.terminate()
    
    # Cleanup
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed StepVideo inference")
    
    # Model paths
    parser.add_argument("--model_dir", type=str, default="./ckpts")
    parser.add_argument("--clip_dir", type=str, default="hunyuan_clip")
    parser.add_argument("--llm_dir", type=str, default="step_llm")
    parser.add_argument("--vae_dir", type=str, default="vae")
    
    # Distributed settings for main model
    parser.add_argument("--ulysses_degree", type=int, default=8)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--tensor_parallel_degree", type=int, default=1)
    
    # Distributed settings for API server
    parser.add_argument("--api_world_size", type=int, default=2,
                        help="Number of GPUs to use for API server")
    parser.add_argument("--api_host", type=str, default="127.0.0.1")
    parser.add_argument("--api_port", type=str, default="8080")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains")
    parser.add_argument("--num_frames", type=int, default=204)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=992)
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=9.0)
    parser.add_argument("--time_shift", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pos_magic", type=str, 
                       default="Ultra HD, HDR video, ambient light, Dolby Atmos, stable shots, smooth motion, realistic details, professional composition, surrealism, natural, vivid, ultra-detailed, crisp.")
    parser.add_argument("--neg_magic", type=str,
                       default="Dark scene, low resolution, bad hands, text, missing fingers, extra fingers, cropped, low quality, grainy, signature, watermark, username, blurry.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine total world size for main model
    world_size = args.ulysses_degree * args.tensor_parallel_degree
    
    # Use torch.multiprocessing to launch distributed processes
    mp.spawn(
        run_distributed_pipeline,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
