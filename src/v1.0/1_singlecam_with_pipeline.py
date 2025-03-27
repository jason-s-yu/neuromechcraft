import gym
import minerl as _
import cv2
import time
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from flygym.vision import Retina

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MineRL simulation with specified action.')
parser.add_argument('ACTION', choices=['random', 'forward'], help='Action to perform in the simulation (either random actions or going forward)', default='forward')
args = parser.parse_args()

out_filename = '1_single_cam_forward_jump.mp4'

# Initialize the test environment; reset the state
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

# Extract first frame shape
height, width, channels = obs['pov'].shape  # e.g. (360, 640, 3)
print(f'Verify obs dims: {height}x{width}x{channels}')

# Set up recorder params
fps = 15.0
max_frames = 600
frame_count = 0
done = False

# Init NMF retina
retina = Retina()

video_writer = None
fourcc = cv2.VideoWriter_fourcc(*'FMP4')

avg_frametime = 0.0

print('Running simulation...')
timestamp_start = time.time()

while not done and frame_count < max_frames:
    frame_start_time = time.time()

    # parse mineRL frame into BGR for openCV to save to vid
    mc_frame_rgb = obs['pov']  # shape (H, W, 3) in RGB
    mc_frame_bgr = cv2.cvtColor(mc_frame_rgb, cv2.COLOR_RGB2BGR) # reframe to BGR

    # Naive approach: single camera duplicated for left and right
    frame = np.ascontiguousarray(obs['pov'])
    fly_vision_left = retina.hex_pxls_to_human_readable(
        retina.raw_image_to_hex_pxls(frame), color_8bit=True
    ).max(axis=-1)
    fly_vision_right = retina.hex_pxls_to_human_readable(
        retina.raw_image_to_hex_pxls(frame), color_8bit=True
    ).max(axis=-1)

    # Show fly representation
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
    axs[0].imshow(fly_vision_left, cmap="gray", vmin=0, vmax=255)
    axs[0].axis("off")
    axs[0].set_title("Left eye")
    axs[1].imshow(fly_vision_right, cmap="gray", vmin=0, vmax=255)
    axs[1].axis("off")
    axs[1].set_title("Right eye")

    # Render the figure to an off-screen buffer (RGBA)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Grab the RGBA buffer from the Agg canvas
    plot_image = np.array(canvas.buffer_rgba())  # shape: (plot_height, plot_width, 4)
    plt.close(fig)

    # Convert RGBA -> BGR for OpenCV
    plot_image_bgr = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)

    plot_height, plot_width, _ = plot_image_bgr.shape

    # resize plot for prettiness
    if plot_width != width:
        scale = width / float(plot_width)
        new_height = int(plot_height * scale)
        plot_image_bgr = cv2.resize(plot_image_bgr, (width, new_height))
    else:
        new_height = plot_height

    # stack Minecraft on top of matplotlib fig
    combined_frame = cv2.vconcat([mc_frame_bgr, plot_image_bgr])

    # init videowriter if not yet based on shape
    if video_writer is None:
        final_height = height + new_height
        video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (width, final_height))
        print(f'Video writer initialized for size ({width}, {final_height})')

    video_writer.write(combined_frame)

    frame_end_time = time.time()
    frame_time_sec = (frame_end_time - frame_start_time)
    print(f'Frame {frame_count+1} time: {frame_time_sec*1000:.3f} ms')

    avg_frametime = (avg_frametime * frame_count + frame_time_sec) / (frame_count + 1)
    frame_count += 1

    # store slow frames for later review
    if frame_time_sec > 0.150:
        filename = f"slow/frame_{frame_count}.png"
        cv2.imwrite(filename, combined_frame)

    # take one step based on the specified action
    action = env.action_space.no_op()
    if args.ACTION == 'forward':
        action['jump'] = 1
        action['forward'] = 1
    elif args.ACTION == 'random':
        action = env.action_space.sample()
        action['inventory'] = 0
    obs, reward, done, info = env.step(action)
    env.render()

timestamp_end = time.time()

if video_writer:
    video_writer.release()
env.close()

elapsed_time = timestamp_end - timestamp_start
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(f"Saved {frame_count} frames to {out_filename} in {elapsed_str}.")
print(f"Average frametime: {avg_frametime*1000:.3f} ms (avg FPS: {1.0/avg_frametime:.2f})")
