import gym
import minerl as _
import numpy as np
import cv2
import time

from flygym.vision import Retina  

# Initialize the test environment; reset the state
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

# Extract the first frame shape for videowriter init
height, width, channels = obs['pov'].shape
overlap = 120
pano_width = width * 3 - overlap * 2
pano_height = height

# Use opencv to write video from RGB data
print('Initializing the video writer.')
fps = 60.0
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_writer = cv2.VideoWriter('1_output.mp4', fourcc, fps, (pano_width, pano_height))

# We want 10 seconds at 60 FPS == 600 frames
max_frames = 600
frame_count = 0
done = False

# Init NMF retina
retina = Retina()

print('Running simulation...')

timestamp_start = time.time()

while not done and frame_count < max_frames:
    center_img = obs['pov']

    # Simulate left eye: turn the camera 90 deg to the left
    action = env.action_space.no_op()
    action['camera'] = [0, -90]           # [pitch_change, yaw_change]: yaw -90 deg
    obs_left, _, _, _ = env.step(action)
    left_img = obs_left['pov']

    # Right eye: turn the camera 180 deg to the right (facing right now)
    action['camera'] = [0, 180]           # turn yaw +180 deg (from -90, this results in +90 relative to original)
    obs_right, _, _, _ = env.step(action)
    right_img = obs_right['pov']

    # Turn the camera back to center (90 deg left from current orientation)
    action['camera'] = [0, -90]
    obs_center, _, _, _ = env.step(action)

    # Now we merge the images together to form a 270 degree FoV with approx 17 overlap per NMF
    overlap = 120
    h, w, c = left_img.shape
    pano = np.zeros((h, w*3 - overlap*2, c), dtype=left_img.dtype)

    pano[:, 0:w] = left_img
    pano[:, w - overlap : w - overlap + w] = center_img
    pano[:, 2*(w - overlap) : 2*(w - overlap) + w] = right_img

    frame = pano

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
    env.render()

    frame_count += 1

timestamp_end = time.time()

# Cleanup
video_writer.release()
env.close()

elapsed_time = timestamp_end - timestamp_start
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print(f"Saved {frame_count} frames into 1_output.mp4 in {elapsed_str}")
