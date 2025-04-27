import argparse
import os
import ffmpeg
from datetime import datetime
from pathlib import Path
from typing import List
import subprocess
import av
import numpy as np
import cv2
import torch
import torchvision
import shutil
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from models.pose_guider import PoseGuider
from models.pose_guider_org import PoseGuiderOrg
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from pipelines.pipeline_pose2vid_long_backup import Pose2VideoPipeline
from utils.util import get_fps, read_frames, save_videos_grid
from utils.frame_interpolation import init_frame_interpolation_model, batch_images_interpolation_tool
from openpose import OpenposeDetector
from utils.mp_utils  import LMKExtractor
from utils.draw_util import FaceMeshVisualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/prompts/inference.yaml')
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("-acc", "--accelerate", action='store_true')
    parser.add_argument("--fi_step", type=int, default=3)
    args = parser.parse_args()

    return args

def read_frames_from_directory(directory_path):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    frame_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(valid_extensions)
    ]
    frame_files.sort()
    frames = [Image.open(frame_file).convert('RGB') for frame_file in frame_files]
    return frames

def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,#config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(noise_latent_channels=320).to(device="cuda", dtype=weight_dtype) # not use cross attention
    # pose_guider = PoseGuiderOrg(
    #     conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)
    # ).to(device="cuda", dtype=weight_dtype)

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
        strict=True,
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
        strict=False, # TONY CHANGED HERE, should be TRUE
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    test_dir = config['test_dir']
    characters_dir = Path(test_dir)
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)    
    
    if args.accelerate:
        frame_inter_model = init_frame_interpolation_model()

    for character_dir in characters_dir.iterdir():
        if character_dir.is_dir():
            character_name = character_dir.name
            motions_dir = character_dir / 'motions'
            for motion_dir in motions_dir.iterdir():
                if motion_dir.is_dir():
                    motion_name = motion_dir.name
                    save_dir_temp = Path(f"output/{date_str}/{save_dir_name}/{character_name}/motions/{motion_name}/ground_truth")
                    save_dir_pred = Path(f"output/{date_str}/{save_dir_name}/{character_name}/motions/{motion_name}/predict")
                    save_dir_temp.mkdir(exist_ok=True, parents=True) 
                    save_dir_pred.mkdir(exist_ok=True, parents=True)
                    ground_truth_dir = motion_dir / 'ground_truth'
                    poses_dir = motion_dir / 'poses'

                    # Get the first image from 'ground_truth' as the reference image
                    ground_truth_images = sorted(ground_truth_dir.glob('*'))
                    for idx, ground_true_path in enumerate(ground_truth_images):
                        shutil.copyfile(ground_true_path, Path(f"{save_dir_temp}/frame_{idx+1}.png"))

                    if not ground_truth_images:
                        print(f"No ground truth images found in {ground_truth_dir}")
                        continue
                    ref_image_path = ground_truth_images[0]

                    # Get all images from 'poses' directory
                    pose_image_paths = sorted(poses_dir.glob('*'))
                    if not pose_image_paths:
                        print(f"No pose images found in {poses_dir}")
                        continue
                    pose_name = motion_name  # You can adjust this if needed
                    # Load the reference image
                    ref_image_pil = Image.open(ref_image_path).convert("RGB")
                    ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
                    ref_image_np = cv2.resize(ref_image_np, (args.W, args.H))

                    # Extract the reference pose
                    ref_pose = Image.open(pose_image_paths[0]).convert("RGB")
                    ref_pose_np = cv2.cvtColor(np.array(ref_pose), cv2.COLOR_RGB2BGR)
                    ref_pose_np = cv2.resize(ref_pose_np, (args.W, args.H))
                    # Load and process pose images
                    pose_images = [Image.open(p).convert("RGB") for p in pose_image_paths]
                    
                    pose_transform = transforms.Compose([
                        transforms.Resize((height, width)),
                        transforms.ToTensor()
                    ])
                    
                    # Determine the number of frames to process
                    args_L = len(pose_images) if args.L is None else args.L

                    # Transform images to tensors
                    pose_tensor_list = [
                        pose_transform(pose_image_pil) for pose_image_pil in pose_images[:args_L]
                    ]
                    pose_list = pose_images[:args_L]

                    # Determine sub-step
                    sub_step = args.fi_step if args.accelerate else 1

                    # Prepare pose list
                    # pose_list = []
                    # for pose_image_pil in pose_images[:args_L:sub_step]:
                    #     pose_image_np = cv2.cvtColor(np.array(pose_image_pil), cv2.COLOR_RGB2BGR)
                    #     pose_image_np = cv2.resize(pose_image_np, (args.W, args.H))
                    #     pose_list.append(pose_image_np)
                    # pose_list = np.array(pose_list)

                    # Get video length
                    video_length = len(pose_list)

                    # Stack tensors
                    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
                    pose_tensor = pose_tensor.transpose(0, 1)          # (c, f, h, w)
                    pose_tensor = pose_tensor.unsqueeze(0)             # (1, c, f, h, w)

                    # Generate video using the pipeline
                    video = pipe(
                        ref_image_pil,
                        pose_list,
                        ref_pose,
                        args.W,
                        args.H,
                        video_length,
                        args.steps,
                        args.cfg,
                        generator=generator,
                    ).videos

                    if config.output_format == "video":
                        if args.accelerate:
                            video = batch_images_interpolation_tool(video, frame_inter_model, inter_frames=args.fi_step-1)
                
                        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
                        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
                            0
                        )  # (1, c, 1, h, w)
                        ref_image_tensor = repeat(
                            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video.shape[2]
                        )

                        video = torch.cat([ref_image_tensor, pose_tensor[:,:,:video.shape[2]], video], dim=0)
                        save_path = f"{save_dir_pred}/{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}_noaudio.mp4"
                        save_videos_grid(
                            video,
                            save_path,
                            n_rows=3,
                            fps=src_fps if args.fps is None else args.fps,
                        )
                    elif config.output_format == "image":
                        # Save each frame as an image
                        num_frames = video.shape[2]  # Assuming video shape is (1, c, f, h, w)
                        for frame_idx in range(num_frames):
                            frame_tensor = video[0, :, frame_idx, :, :]  # (c, h, w)
                            frame_array = frame_tensor.cpu().numpy()
                            frame_array = np.transpose(frame_array, (1, 2, 0))  # (h, w, c)
                            frame_array = (frame_array * 255).astype(np.uint8)
                            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                            frame_filename = save_dir_pred / f"frame_{(1+frame_idx)}.png"
                            cv2.imwrite(str(frame_filename), frame_array)
                    else:
                        print(f"Non support output format")

                    print(f"Saved frames to {save_dir_pred}")


if __name__ == "__main__":
    main()
