import gym
import minerl as _
import cv2
import time
import numpy as np
import argparse
import os
import shutil

from flygym import Fly
from flygym.vision import Retina

# arg parser
parser = argparse.ArgumentParser(description='Run MineRL simulation with specified action.')
parser.add_argument('ACTION', choices=['random', 'forward'], default='forward', nargs='?',
                    help='Action to perform in the simulation (either random actions or going forward)')
args = parser.parse_args()

out_filename = '1_single_cam_forward_jump.mp4'

# rm slow dir
if os.path.exists('slow'):
    print('Removing old slow frames...')
    shutil.rmtree('slow')
os.makedirs('slow', exist_ok=True)

# MineRL initialization
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

# Extract first frame shape
height, width, channels = obs['pov'].shape  # (64, 64, 3)
print(f'Verify obs dims: {height}x{width}x{channels}')

# videowriter config
fps = 15.0
max_frames = 600
frame_count = 0
done = False

fly = Fly(enable_vision=True)
retina = fly.retina

print(f'Retina initialized with: ncols: {retina.ncols}, nrows: {retina.nrows}')

# using 'mp4v' to avoid FMP4 fallback warning
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None

avg_frametime = 0.0

print("=" * 70)
print(f'Running simulation with ACTION = {args.ACTION} for {max_frames} frames')
print("=" * 70)
timestamp_start = time.time()

while not done and frame_count < max_frames:
    frame_start_time = time.time()

    # convert Minecraft frame to BGR for OpenCV
    mc_frame_rgb = obs['pov']  # shape is (H, W, 3) in RGB
    mc_frame_bgr = cv2.cvtColor(mc_frame_rgb, cv2.COLOR_RGB2BGR) # for openCV display

    # compute fly vision (grayscale) and convert to BGR
    frame_conv_start = time.time()
    hex_pxls = retina.raw_image_to_hex_pxls(cv2.resize(mc_frame_rgb, (retina.ncols, retina.nrows)))

    fly_vision_rgb = retina.hex_pxls_to_human_readable(
        hex_pxls,
        color_8bit=True
    )

    print('hex_pixels shape:', hex_pxls.shape) # should be (721, 2)
    print("fly_vision_rgb shape:", fly_vision_rgb.shape) # should be (512, 450, 2)

    fly_vision_gray = fly_vision_rgb.max(axis=-1)

    frame_conv_end = time.time()
    frame_conv_total = frame_conv_end - frame_conv_start

    fly_vision_bgr = cv2.cvtColor(fly_vision_gray, cv2.COLOR_GRAY2BGR)

    # ensure same height for side-by-side. 
    # if their heights differ, consider resizing one or both to match.
    # e.g. if fly_vision_bgr is not 64 px tall, adjust here:
    # fly_vision_bgr = cv2.resize(fly_vision_bgr, (width, height), interpolation=cv2.INTER_NEAREST)

    # combine side-by-side
    if (fly_vision_bgr.shape[0] != height) or (fly_vision_bgr.shape[1] != width):
        fly_vision_bgr = cv2.resize(
            fly_vision_bgr,
            (width, height),  # Match Minecraft's (64,64)
            interpolation=cv2.INTER_NEAREST
        )
    
    combined_bgr = np.hstack((mc_frame_bgr, fly_vision_bgr))

    # upscale for visibility
    upscale_factor = 4
    combined_bgr_up = cv2.resize(combined_bgr, None, 
                                 fx=upscale_factor, 
                                 fy=upscale_factor, 
                                 interpolation=cv2.INTER_NEAREST)

    # initialize the video writer on first frame (once we know final size)
    video_writer_start = time.time()
    if video_writer is None:
        final_height, final_width = combined_bgr_up.shape[:2]
        video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (final_width, final_height))
        print(f'Video writer initialized with size ({final_width}, {final_height})')
        tmp_end_time = time.time()

    # write current frame
    video_writer.write(combined_bgr_up)
    video_writer_end = time.time()
    video_writer_time = video_writer_end - video_writer_start

    # update timing
    frame_end_time = time.time()
    frame_time_sec = frame_end_time - frame_start_time
    print(f'Frame {(frame_count+1):3d} time: {frame_time_sec*1000:.3f} ms [flygym pipeline time: {(frame_conv_total)*1000:.3f} ms, frame write time: {(video_writer_time)*1000:.3f} ms]')

    # store slow frames
    if frame_time_sec > 0.150:
        slow_frame_name = f"slow/frame_{frame_count+1}.png"
        cv2.imwrite(slow_frame_name, combined_bgr_up)

    # update stats
    avg_frametime = (avg_frametime * frame_count + frame_time_sec) / (frame_count + 1)
    frame_count += 1

    action = env.action_space.noop()
    if args.ACTION == 'forward':
        action['jump'] = 1
        action['forward'] = 1
    elif args.ACTION == 'random':
        action = env.action_space.sample()
        # zero out inventory usage
        action['inventory'] = 0

    obs, reward, done, info = env.step(action)

print('=' * 50)
print('Simulation finished. Attempting to finalize files.')
print('=' * 50)

timestamp_end = time.time()

if video_writer is not None:
    video_writer.release()
env.close()

elapsed_time = timestamp_end - timestamp_start
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(f"Saved {frame_count} frames to {out_filename} in {elapsed_str}.")
print(f"Average frametime: {avg_frametime*1000:.3f} ms (avg FPS: {1.0/avg_frametime:.2f})")
