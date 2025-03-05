import gym
import minerl as _
import cv2
import time

from flygym.vision import Retina  

# Initialize the test environment; reset the state
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

# Extract the first frame shape for videowriter init
height, width, channels = obs['pov'].shape  # e.g., (360, 640, 3)

# Use opencv to write video from RGB data
print('Initializing the video writer.')
fps = 60.0
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_writer = cv2.VideoWriter('0_output.mp4', fourcc, fps, (width, height))

# We want 10 seconds at 60 FPS == 600 frames
max_frames = 600
frame_count = 0
done = False

# Init NMF retina
retina = Retina()

print('Running simulation...')

timestamp_start = time.time()

while not done and frame_count < max_frames:
    frame_start_time = time.time()
    frame = obs['pov']  # (height, width, 3) in RGB

    # Convert RGB -> BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Write frame to video
    video_writer.write(frame_bgr)

    # def some_fly_decision_model(fly_vision): ...
    # frame = np.ascontiguousarray(obs['pov']) # also import numpy as np
    # fly_vision_left = retina.raw_image_to_hex_pxls(frame)
    # fly_vision_right = retina.raw_image_to_hex_pxls(frame)
    # fly_vision = np.stack([fly_vision_left, fly_vision_right], axis=0)
    # decision = some_fly_decision_model(fly_vision)
    # action = convert_decision_to_minecraft_action(decision)

    # Sit still
    action = env.action_space.no_op()
    action["ESC"] = 0 # setting esc 1 will quit the environment
    obs, reward, done, info = env.step(action)
    # print(info)
    env.render()
    frame_end_time = time.time()

    elapsed = (frame_end_time - frame_start_time)
    print('Frame time: {:.2f} ms'.format(elapsed * 1000))

    frame_count += 1

timestamp_end = time.time()

# Cleanup
video_writer.release()
env.close()

elapsed_time = timestamp_end - timestamp_start
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print(f"Saved {frame_count} frames into 0_output.mp4 in {elapsed_str}")
