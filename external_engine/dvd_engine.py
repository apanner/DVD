#!/usr/bin/env python3
"""
DVD (Deterministic Video Depth) Engine Adapter
Matches VDA job_data interface for Cinesculpt pipeline integration.
Supports image sequences and outputs EXR + optional MP4.
"""

import os
import re
import sys
from pathlib import Path

# Add parent DVD root for imports
current_dir = Path(__file__).parent
dvd_root = current_dir.parent
if str(dvd_root) not in sys.path:
    sys.path.insert(0, str(dvd_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

# Enable OpenEXR for output
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def read_image_sequence(input_pattern, first_frame, last_frame, frame_padding=4):
    """
    Load image sequence from pattern (e.g., path/file.%04d.jpg).
    Returns (video_tensor [1,T,C,H,W], orig_size (H,W)).
    """
    frames = []
    for frame_num in range(first_frame, last_frame + 1):
        frame_path = re.sub(
            r'%0?(\d+)d',
            lambda m: f"{frame_num:0{int(m.group(1))}d}",
            input_pattern
        )
        if os.path.exists(frame_path):
            img = cv2.imread(frame_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        else:
            print(f"[WARN] Missing frame: {frame_path}")
    
    if not frames:
        raise FileNotFoundError(f"No frames found for pattern: {input_pattern}")
    
    video_np = np.stack(frames)
    orig_H, orig_W = video_np.shape[1], video_np.shape[2]
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), (orig_H, orig_W)


def read_video(video_path):
    """Load video file. Returns (video_tensor [1,T,C,H,W], fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), fps


# UHD maximum dimensions (same as VDA engine)
UHD_MAX_WIDTH = 4096
UHD_MAX_HEIGHT = 2160


def align_to_16(val):
    """Round up to nearest multiple of 16 (DVD pipeline requirement)."""
    return (val + 15) // 16 * 16


def resize_frames_to_uhd_max(video_tensor):
    """
    VDA-style smart resize: process at native resolution up to UHD max (4096x2160).
    - If input > UHD: scale DOWN to fit (aspect ratio preserved)
    - If input <= UHD: use as-is, align to 16 (pad if needed)
    Returns: (resized_tensor, orig_size, processed_size)
    """
    B, T, C, H, W = video_tensor.shape
    orig_size = (H, W)
    
    # Check if resize is needed (exceeds UHD)
    needs_resize = W > UHD_MAX_WIDTH or H > UHD_MAX_HEIGHT
    
    if needs_resize:
        scale_w = UHD_MAX_WIDTH / W
        scale_h = UHD_MAX_HEIGHT / H
        scale = min(scale_w, scale_h)
        new_W = int(round(W * scale))
        new_H = int(round(H * scale))
        new_W = min(align_to_16(new_W), UHD_MAX_WIDTH)
        new_H = min(align_to_16(new_H), UHD_MAX_HEIGHT)
        print(f"[DVD RESIZE] Input {W}x{H} exceeds UHD -> scaling to {new_W}x{new_H}")
    else:
        # Within UHD: align to 16 (pad if needed for DVD pipeline)
        new_W = align_to_16(W)
        new_H = align_to_16(H)
        if new_W == W and new_H == H:
            return video_tensor, orig_size, orig_size
        print(f"[DVD RESIZE] Input {W}x{H} -> aligning to {new_W}x{new_H} (divisible by 16)")
    
    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(new_H, new_W), mode="bilinear", align_corners=False)
    resized = resized.view(B, T, C, new_H, new_W)
    processed_size = (new_H, new_W)
    return resized, orig_size, processed_size


def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


def pad_time_mod4(video_tensor):
    """Pad temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    if mask is None:
        mask = np.ones_like(ref_frames)
    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)
    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0
    return scale, shift


def generate_depth_sliced(model, input_rgb, window_size=81, overlap=21):
    """Run DVD inference with sliding window."""
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    depth_res_list = []
    
    for start, end in tqdm(depth_windows, desc="DVD Inferencing"):
        _input_rgb_slice = input_rgb[:, start:end]
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]
        
        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones([B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        depth_res_list.append(outputs['depth'][:, :origin_T])
    
    # Overlap alignment
    depth_list_aligned = None
    prev_end = None
    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue
        curr_start = start
        real_overlap = prev_end - curr_start
        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]
            scale, shift = compute_scale_and_shift(curr_frames, ref_frames)
            scale = np.clip(scale, 0.7, 1.5)
            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]
            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap, aligned_t[:, real_overlap:]],
                axis=1
            )
        else:
            depth_list_aligned = np.concatenate([depth_list_aligned, t], axis=1)
        prev_end = end
    
    return depth_list_aligned[:, :T]


def save_depth_exr(depth_np, exr_output_dir, first_frame, floating_point='float32'):
    """Save depth to EXR sequence."""
    os.makedirs(exr_output_dir, exist_ok=True)
    try:
        import OpenEXR
        import Imath
    except ImportError:
        # Fallback: use cv2 if OpenEXR not available
        for i in range(depth_np.shape[0]):
            frame_num = first_frame + i
            depth_frame = depth_np[i, :, :, 0].astype(np.float32)
            exr_path = os.path.join(exr_output_dir, f"depth.{frame_num:04d}.exr")
            cv2.imwrite(exr_path, depth_frame)
        return depth_np.shape[0]
    
    count = 0
    for i in range(depth_np.shape[0]):
        frame_num = first_frame + i
        depth_frame = depth_np[i, :, :, 0].astype(np.float32)
        h, w = depth_frame.shape
        exr_path = os.path.join(exr_output_dir, f"depth.{frame_num:04d}.exr")
        header = OpenEXR.Header(w, h)
        header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
        exr = OpenEXR.OutputFile(exr_path, header)
        exr.writePixels({'R': depth_frame.tobytes()})
        exr.close()
        count += 1
    return count


class DVDVideoDepthEngine:
    """
    DVD engine matching VDA job_data interface.
    VDA-style smart resize: process at native resolution up to UHD (4096x2160).
    Downscale only if input exceeds UHD; otherwise use as-is (align to 16).
    """
    
    DIVISION_FACTOR = 16  # DVD pipeline requirement
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or str(dvd_root / "ckpt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load DVD model from ckpt directory."""
        from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule
        from accelerate import Accelerator
        
        config_path = os.path.join(self.model_path, "model_config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"DVD config not found: {config_path}")
        
        yaml_args = OmegaConf.load(config_path)
        accelerator = Accelerator()
        self.model = WanTrainingModule(
            accelerator=accelerator,
            model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
            trainable_models=None,
            use_gradient_checkpointing=False,
            lora_rank=yaml_args.lora_rank,
            lora_base_model=yaml_args.lora_base_model,
            args=yaml_args,
        )
        ckpt_path = os.path.join(self.model_path, "model.safetensors")
        state_dict = load_file(ckpt_path, device="cpu")
        dit_state_dict = {k.replace("pipe.dit.", ""): v for k, v in state_dict.items() if "pipe.dit." in k}
        self.model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
        self.model.merge_lora_layer()
        self.model = self.model.to(self.device)
        return True
    
    def process_video_original(self, job_data):
        """
        Process video/image sequence - matches VDA job_data interface.
        Returns: {'depth_frames': N, 'output_path': exr_dir}
        """
        input_video = job_data["input_video"]
        exr_output_dir = job_data["exr_output_dir"]
        first_frame = int(job_data['first_frame'])
        last_frame = int(job_data['last_frame'])
        floating_point = job_data.get('floating_point', 'float32')
        depth_mp4_dir = job_data.get('depth_mp4_dir')
        create_depth_vis_mp4 = job_data.get('create_depth_vis_mp4', True)
        
        # DVD-specific params
        window_size = job_data.get('dvd_window_size', 81)
        overlap = job_data.get('dvd_overlap', 21)
        
        # Load input at original resolution
        if input_video.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_tensor, fps = read_video(input_video)
        else:
            frame_padding = int(job_data.get('frame_padding', 4))
            input_tensor, _ = read_image_sequence(input_video, first_frame, last_frame, frame_padding)
            fps = 24
        
        # VDA-style smart resize: process at native up to UHD max (4096x2160)
        # Downscale only if exceeds UHD; otherwise use as-is (align to 16 for DVD pipeline)
        input_tensor, orig_size, processed_size = resize_frames_to_uhd_max(input_tensor)
        input_tensor = input_tensor.to(self.device)
        print(f"[DVD] Processing at {processed_size[1]}x{processed_size[0]} (UHD-cap: {UHD_MAX_WIDTH}x{UHD_MAX_HEIGHT})")
        
        # Load model if not loaded
        if self.model is None:
            self.load_model()
        
        # Run inference - depth output at processed resolution (same as VDA)
        depth = generate_depth_sliced(self.model, input_tensor, window_size, overlap)[0]
        # Output at processed size (max quality) - no resize_back, like VDA engine
        
        # Save EXR
        count = save_depth_exr(depth, exr_output_dir, first_frame, floating_point)
        
        # Optional: save MP4 preview
        if create_depth_vis_mp4 and depth_mp4_dir:
            os.makedirs(depth_mp4_dir, exist_ok=True)
            d_min, d_max = depth.min(), depth.max()
            vis_depth = (depth - d_min) / (d_max - d_min + 1e-8)
            try:
                from diffsynth import save_video
                mp4_path = os.path.join(depth_mp4_dir, "depth_vis.mp4")
                save_video(vis_depth, mp4_path, fps=fps, quality=6, grayscale=True)
            except Exception as e:
                print(f"[WARN] Could not save MP4 preview: {e}")
        
        return {'depth_frames': count, 'output_path': exr_output_dir}
