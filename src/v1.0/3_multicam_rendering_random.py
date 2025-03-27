"""
Now suppose we want to extend the FOV from one cam (which provides horizontal FOV of about 103 degrees).

We can use 4 cameras to cover >360 degrees of view (about 430)
and then stitch them together to produce the 17 degree front middle overlap, 
ignoring the excess ~100 degrees on the rear.

We can do this by commanding the agent to turn, capture the observation frames, and then stitch them together.

This is obviously not ideal as far as real time processing goes (there could be noticeable input delay)
because the agent has to turn and capture 4 distinct frames nearly instantaneously.
"""

import gym
import minerl as _
import numpy as np
import cv2
import time

out_filename = '2_output.mp4'

# Agent obs pov frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Each camera ~103 deg horizontal at Minecraft's default FOV=70 (vertical)
CAM_HFOV = 103.0
deg_per_px = CAM_HFOV / FRAME_WIDTH  # yields ~0.161 deg/px

# Overlaps within each hemisphere (62.5 deg => ~388 px)
# (That's the big overlap between "far" and "mid" camera in each hemisphere.)
overlap_px_hemisphere = int(round(62.5 / deg_per_px))  # ~388

# Hemisphere mosaic from 2 cameras => 640 + 640 - overlap_px => 892 wide
hemisphere_width = FRAME_WIDTH*2 - overlap_px_hemisphere  # 892

# The final mosaic is left hemisphere + right hemisphere side by side,
# with a 17 deg front overlap physically duplicated (not merged).
# So total width = 892 + 892 = 1784
final_width = hemisphere_width * 2
final_height = FRAME_HEIGHT

print(f"Overlap (intra-hemisphere) = {overlap_px_hemisphere} px")
print(f"Hemisphere mosaic size = {hemisphere_width} x {FRAME_HEIGHT}")
print(f"Final mosaic size = {final_width} x {final_height}")

# Yaw angles (approx camera centers):
# These 4 cameras collectively span ~[-135 deg, +135 deg],
# with about 17 deg duplicated around the front center.
yaw_left_far  = -83.5  # covers ~[-135, -32]
yaw_left_mid  = -43.0  # covers ~[-94.5, +8.5]
yaw_right_mid = +43.0  # covers ~[-8.5, +94.5]
yaw_right_far = +83.5  # covers ~[+32, +135]

# We'll trim a chunk of the far_left image's right side and far_right image's left side
# to remove the "rear overlap" that doesn't add coverage beyond the 270 drosophilia view.
# For instance, each far camera covers 103 deg, but we only need approx. half of that
# to reach the boundary near +- 32° in each hemisphere.
# Each "rear trim" can be ~ (103/2) -> ~51.5 deg => ~320 px. Or specifically, from geometry:
# The far_left is centered at -83.5 => covers [-135..-32], we don't need beyond -32.
# That's about 103 deg total, so there's not a huge chunk we must forcibly remove,
# but let's do an explicit slice so it won't intrude on the mid camera region.
# We'll pick a safe number around 32 deg from center => 51.5 deg => ~320 px. 
# (You can tweak as needed.)

TRIM_DEG_FAR = 50.0  # about how many degrees to cut off from the far camera
TRIM_PX_FAR  = int(round(TRIM_DEG_FAR / deg_per_px))  # ~311 px

print(f"Trimming ~{TRIM_DEG_FAR} deg => ~{TRIM_PX_FAR} px from each far camera.")


current_yaw = 0.0

def turn_to(env, yaw_target):
    """
    Perform a relative turn to the target yaw, from a global current_yaw.
    Then, step once in the environment.
    Returns (obs, done)
    """
    global current_yaw
    delta = yaw_target - current_yaw
    action = env.action_space.no_op()
    action['camera'] = [0, delta]
    obs, _, done, _ = env.step(action)
    current_yaw = yaw_target
    return obs, done

def stitch_hemisphere(img_far, img_mid, overlap_px, trim_left=0, trim_right=0):
    """
    Stitches two 360x640 frames horizontally, subtracting 'overlap_px' from the join region.
    Essentially, this builds each hemisphere (partial panorama or mosaic) from two cameras.
    Final shape: (360, 640*2 - overlap_px, 3).

    We added trim_left and trim_right to handle "rear overlap" 
    so we can discard the unused portion of the far camera from either side.
    """

    w_out = FRAME_WIDTH*2 - overlap_px
    mosaic = np.zeros((FRAME_HEIGHT, w_out, 3), dtype=np.uint8)

    # 1) Trim the "far" camera's coverage from the right side
    #    so it doesn't intrude on the center region.
    #    We'll remove TRIM_PX_FAR from the right edge.
    #    (If trim_left > 0, it also removes from the left.)
    far_cropped = img_far[:, trim_left : FRAME_WIDTH - trim_right]

    far_trimmed_width = far_cropped.shape[1]
    if far_trimmed_width < 0:
        far_trimmed_width = 0

    # 2) Place the far-cropped portion at x=0..(far_trimmed_width-1)
    mosaic[:, 0:far_trimmed_width] = far_cropped

    # 3) Overlap with "mid" camera
    #    The overlap is 'overlap_px' in pixel space.
    start_mid = far_trimmed_width - overlap_px
    if start_mid < 0:
        start_mid = 0  # just to avoid negative indexing if we heavily trimmed

    mosaic[:, start_mid : start_mid+FRAME_WIDTH] = img_mid

    return mosaic


# MineRL initialization
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()
done = False

fps = 10
fourcc = cv2.VideoWriter_fourcc(*'FMP4')
video_writer = cv2.VideoWriter(out_filename, fourcc, fps, (final_width, final_height))

# Main loop
max_frames = 300
frame_count = 0
timestamp_start = time.time()

while not done and frame_count < max_frames:
    obs_fl, done = turn_to(env, yaw_left_far)
    img_fl = obs_fl['pov']

    obs_lm, done = turn_to(env, yaw_left_mid)
    img_lm = obs_lm['pov']

    obs_rm, done = turn_to(env, yaw_right_mid)
    img_rm = obs_rm['pov']

    obs_fr, done = turn_to(env, yaw_right_far)
    img_fr = obs_fr['pov']

    # reset cam to 0 deg
    turn_to(env, 0.0)

    # Build left hemisphere from far_left + left_mid, trimming far_left from the right side
    left_hemi = stitch_hemisphere(
        img_far=img_fl,
        img_mid=img_lm,
        overlap_px=overlap_px_hemisphere,
        trim_left=0,
        trim_right=TRIM_PX_FAR
    )

    # Build right hemisphere from far_right + right_mid,
    # but we want to trim the far_right camera from the *left* side
    right_hemi = stitch_hemisphere(
        img_far=img_fr,
        img_mid=img_rm,
        overlap_px=overlap_px_hemisphere,
        trim_left=TRIM_PX_FAR,
        trim_right=0
    )

    # Merge left and right hemispheres together
    # retaining the 17 degree overlap
    # we can just split down the middle to apply for left and right eyeballs
    final_mosaic = np.zeros((final_height, final_width, 3), dtype=np.uint8)
    final_mosaic[:, 0:hemisphere_width] = left_hemi
    final_mosaic[:, hemisphere_width : hemisphere_width*2] = right_hemi

    frame_bgr = cv2.cvtColor(final_mosaic, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

    action = env.action_space.no_op()
    obs, reward, done, info = env.step(action)
    env.render()
    frame_count += 1

timestamp_end = time.time()
video_writer.release()
env.close()

elapsed_sec = timestamp_end - timestamp_start
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
print(f"Saved {frame_count} frames in {elapsed_str} to {out_filename}.")
