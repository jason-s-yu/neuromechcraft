import gym
import minerl as _
import numpy as np
import cv2
import time
import math
import logging
from tqdm import tqdm

# Progress bar config
# Define a handler that uses tqdm.write to ensure logs print above the bar
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        # Use default terminator from logging.StreamHandler which is '\n'
        self.terminator = '\n'

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Logger config
# Get our specific logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default level - hides DEBUG messages unless changed

# Create and configure the custom Tqdm handler
tqdm_handler = TqdmLoggingHandler()
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
tqdm_handler.setFormatter(formatter)

# Add the handler to our logger
logger.addHandler(tqdm_handler)

# Prevent propagation to avoid potential double logging if root logger is configured
logger.propagate = False

# CONFIG FLAGS
USE_CONCATENATION_ALWAYS = True # Set True to skip stitching attempt, faster and avoids warnings on low-res

# Target total horizontal FOV (degrees) for the NMF simulation
TARGET_FOV = 270.0
# Estimated FOV of a single MineRL v0.4.4 capture (64x64)
SINGLE_CAPTURE_HFOV = 70.0
# Overlap between captured frames (degrees) - helps ensure coverage
STITCH_OVERLAP = 20.0
# NMF specific: Approximate overlap between the final left/right eye views
NMF_BINOCULAR_OVERLAP_DEG = 17.0

# Visualization config
DISPLAY_SCALE_FACTOR = 4 # Upscale the final output window
ENABLE_HEX_VISUALIZATION = True # Set to False to disable retina vis
HEX_GRID_RADIUS = 3 # Controls size of hexagons in visualization
HEX_GRID_COLS = 12 # Approximate number of columns for simulated retina hex grid
HEX_GRID_ROWS = 10 # Approximate number of rows

# Obstacle avoidance / decision config
BRIGHTNESS_THRESHOLD = 50 # Avg pixel value below which area is considered 'dark' (0-255)
TURN_MAGNITUDE = 15.0 # Degrees to turn per step for avoidance
ALWAYS_MOVE_FORWARD = True
ALWAYS_SPRINT = True
ALWAYS_JUMP = True # Note: Jump might need specific timing in MineRL

# Video output config
SAVE_VIDEO = True # Set to True to save the output visualization
OUTPUT_FILENAME = "2_fakefov.mp4" # Default output video filename
OUTPUT_FPS = 10.0 # Target FPS for the output video (actual runtime FPS will vary)

# Calculate parameters
effective_fov_per_capture = SINGLE_CAPTURE_HFOV - STITCH_OVERLAP
if effective_fov_per_capture <= 0:
    logger.error(f"STITCH_OVERLAP ({STITCH_OVERLAP}) must be less than SINGLE_CAPTURE_HFOV ({SINGLE_CAPTURE_HFOV})")
    raise ValueError("STITCH_OVERLAP must be less than SINGLE_CAPTURE_HFOV")

num_additional_captures_needed = np.ceil((TARGET_FOV - SINGLE_CAPTURE_HFOV) / max(1.0, effective_fov_per_capture)) # Avoid division by zero
num_captures = int(1 + num_additional_captures_needed)
if num_captures % 2 == 0: num_captures += 1 # Ensure odd number for symmetry

yaw_per_capture_step = effective_fov_per_capture
total_yaw_span = (num_captures - 1) * yaw_per_capture_step
initial_yaw_turn = - (num_captures // 2) * yaw_per_capture_step
reset_yaw_turn = -initial_yaw_turn # Theoretical reset needed if done manually

# Log configuration details at INFO level
logger.info("--- Configuration ---")
logger.info(f"Processing Mode: {'Force Concatenation' if USE_CONCATENATION_ALWAYS else 'Attempt Stitching -> Fallback Concatenation'}")
logger.info(f"MineRL Version: Expected v0.4.4 (using 64x64 pov)")
logger.info(f"Target FOV: {TARGET_FOV} deg")
logger.info(f"Single Capture HFOV: {SINGLE_CAPTURE_HFOV} deg")
logger.info(f"Stitch Overlap: {STITCH_OVERLAP} deg")
logger.info(f"NMF Binocular Overlap: {NMF_BINOCULAR_OVERLAP_DEG} deg")
logger.info(f"Calculated Captures Needed: {num_captures}")
logger.info(f"Yaw Turn Per Capture Step: {yaw_per_capture_step:.2f} deg")
logger.info(f"Initial Centering Yaw: {initial_yaw_turn:.2f} deg")
logger.info(f"Hex Grid Dimensions: {HEX_GRID_ROWS} rows x {HEX_GRID_COLS} cols")
logger.info(f"Video Saving Enabled: {SAVE_VIDEO}")
if SAVE_VIDEO:
    logger.info(f"Output Filename: {OUTPUT_FILENAME}")
    logger.info(f"Output Target FPS: {OUTPUT_FPS}")
logger.info("--------------------")


def turn_agent(env, yaw_degrees):
    """Applies a yaw turn action and updates the environment's current observation."""
    action = env.action_space.noop()
    action['camera'] = np.array([0, yaw_degrees], dtype=np.float32)
    obs = None
    done = False
    try:
        obs, _, done, _ = env.step(action)
        if obs is not None:
            env.current_obs = obs # Update shared observation state
        else:
            logger.warning("turn_agent received None observation from env.step. Episode might have ended.")
            # Keep previous obs if current is None, let done flag handle termination
            done = True # Assume done if observation is None after an action
        return obs, done
    except Exception as e:
        logger.error(f"Error during env.step() in turn_agent: {e}", exc_info=True)
        return env.current_obs, True # Assume done if error occurs

def simulate_wide_fov_mineRL(env, num_captures, yaw_per_step, initial_turn):
    """
    Simulates a wide FOV by rotating, capturing, and optionally attempting stitching
    before falling back to concatenation (or forces concatenation).
    Performs multiple env.step calls internally for camera turns.
    Resets camera orientation at the end.
    Returns: stitched/concatenated image (RGB), list of raw frames, done flag.
    """
    frames = []
    done = False

    if not hasattr(env, 'current_obs') or env.current_obs is None:
        logger.warning("env.current_obs is missing or None at start of simulate_wide_fov. Attempting initial step.")
        action = env.action_space.noop()
        try:
            obs, _, done, _ = env.step(action)
            if obs is not None:
                env.current_obs = obs
            else:
                logger.error("Received None observation on initial step in simulate_wide_fov.")
                done = True # If still none, something is wrong
        except Exception as e:
            logger.error(f"Error during initial step in simulate_wide_fov: {e}", exc_info=True)
            done = True
        if env.current_obs is None or done:
            logger.error("Could not get initial observation or episode ended immediately.")
            return None, [], True

    # Initial turn
    logger.debug(f"Initial turn: {initial_turn:.1f} degrees")
    _, done = turn_agent(env, initial_turn)
    if done:
        logger.warning("Episode ended during initial turn for FOV simulation.")
        return None, [], done

    # Capture sequence
    total_yaw_turned_capture = 0
    for i in range(num_captures):
        if done: break # Don't continue if episode ended during turns

        if env.current_obs is None:
            logger.warning(f"env.current_obs became None before capture {i+1}. Stopping capture sequence.")
            done = True # Treat as episode end
            break

        current_pov = env.current_obs.get('pov')
        if current_pov is None or current_pov.size == 0:
            logger.warning(f"Got None or empty POV at capture {i+1}. Skipping frame.")
            # Attempt to continue turning to maintain sequence? Or break? Let's skip frame.
        else:
            frames.append(current_pov)
            logger.debug(f"Captured frame {i+1}/{num_captures}")

        # Turn for the next capture, unless this was the last one
        if i < num_captures - 1:
            logger.debug(f"Turning for next capture: +{yaw_per_step:.1f} degrees")
            _, turn_done = turn_agent(env, yaw_per_step)
            if not turn_done:
                total_yaw_turned_capture += yaw_per_step
            else:
                logger.warning("Episode ended during FOV capture sequence turn.")
                done = True # Mark episode as done
                # Break here, as further captures/reset are meaningless
                break

    # Reset camera orientation
    # Calculate the turn needed to return to the orientation *before* this function started
    reset_turn_amount = -(initial_turn + total_yaw_turned_capture)
    if not done and abs(reset_turn_amount) > 1e-3: # Only reset if not done and a net turn occurred
        logger.debug(f"Resetting camera by: {reset_turn_amount:.1f} degrees")
        _, reset_done = turn_agent(env, reset_turn_amount)
        done = done or reset_done # If reset causes done, update flag

    # Image stitching/concatenation
    stitched_image_rgb = None
    if len(frames) < 1:
        logger.error("No valid frames captured during FOV simulation.")
        # Attempt to return the *last known good* single POV if available
        if hasattr(env, 'current_obs') and env.current_obs and 'pov' in env.current_obs and env.current_obs['pov'] is not None:
            logger.warning("Returning last valid single POV image instead of wide FOV.")
            stitched_image_rgb = env.current_obs['pov']
        else:
            logger.error("No frames captured and no current POV available.")
            # done should already be true if we reached here without frames/obs
            return None, [], True # No image available
    elif len(frames) == 1:
        logger.info("Only one frame captured, using it directly.")
        stitched_image_rgb = frames[0]
    else:
        logger.debug(f"Processing {len(frames)} captured frames (shape: {frames[0].shape})...")
        stitched_ok = False

        # Conditionally attempt stitching
        if not USE_CONCATENATION_ALWAYS:
            try:
                logger.debug("Attempting OpenCV stitching...")
                # Ensure frames are in BGR format for OpenCV Stitcher
                frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
                stitcher = cv2.Stitcher_create()
                status, stitched_image_bgr = stitcher.stitch(frames_bgr)
                if status == cv2.Stitcher_OK and stitched_image_bgr is not None:
                    logger.debug("Stitching successful.")
                    stitched_image_rgb = cv2.cvtColor(stitched_image_bgr, cv2.COLOR_BGR2RGB)
                    stitched_ok = True
                elif stitched_image_bgr is None:
                    logger.warning(f"Stitching returned status {status} but image is None. Falling back.")
                else:
                    logger.warning(f"Stitching failed with status code: {status}. Falling back.")
            except cv2.error as e:
                logger.warning(f"OpenCV error during stitching (likely low resolution/features): {e}. Falling back.")
            except Exception as e:
                logger.error(f"Unexpected error during stitching: {e}", exc_info=True)
                logger.warning("Falling back to concatenation after unexpected error.")
        else:
            logger.debug("USE_CONCATENATION_ALWAYS is True, skipping stitching attempt.")

        # Use Concatenation if stitching wasn't attempted, failed, or isn't desired
        if not stitched_ok:
            if USE_CONCATENATION_ALWAYS:
                logger.debug("Using horizontal concatenation (forced).")
            else:
                logger.info("Using horizontal concatenation as fallback.") # Info level if it's a fallback

            # Ensure all frames have the same height before concatenating
            min_height = min(f.shape[0] for f in frames)
            if any(f.shape[0] != min_height for f in frames):
                logger.debug(f"Resizing frames to minimum height ({min_height}) for concatenation.")
                frames_rgb_resized = [cv2.resize(f, (f.shape[1], min_height), interpolation=cv2.INTER_NEAREST) for f in frames]
            else:
                frames_rgb_resized = frames # No resize needed if heights match

            try:
                stitched_image_rgb = cv2.hconcat(frames_rgb_resized)
                logger.debug(f"Concatenation successful. Final shape: {stitched_image_rgb.shape}")
            except Exception as e:
                logger.error(f"Error during concatenation: {e}", exc_info=True)
                logger.warning("Concatenation failed. Returning first captured frame as last resort.")
                if frames: stitched_image_rgb = frames[0] # Fallback to first frame

    # Final check if image processing somehow resulted in None
    if stitched_image_rgb is None and frames:
        logger.warning("Image processing resulted in None, returning first captured frame.")
        stitched_image_rgb = frames[0]
    elif stitched_image_rgb is None:
        logger.error("Could not produce any output image from FOV simulation.")
        # Ensure done is true if we have no image
        done = True

    return stitched_image_rgb, frames, done

def get_hexagon_vertices(center_x, center_y, radius):
    """ Calculates the 6 vertices of a hexagon. """
    vertices = []
    for i in range(6):
        angle_deg = 60 * i - 30 # Start angle to have flat top/bottom
        angle_rad = math.pi / 180 * angle_deg
        x = center_x + radius * math.cos(angle_rad)
        y = center_y + radius * math.sin(angle_rad)
        vertices.append((int(round(x)), int(round(y)))) # Round for pixel coords
    return np.array(vertices, dtype=np.int32)

def draw_simulated_retina(canvas, center_x, center_y, grid_rows, grid_cols, hex_radius, brightness_values):
    """ Draws a hexagonal grid representing a retina on the canvas. """
    hex_width = hex_radius * 2
    hex_height = math.sqrt(3) * hex_radius
    val_idx = 0
    total_hexes = grid_rows * grid_cols

    if not brightness_values or len(brightness_values) != total_hexes:
        logger.warning(f"Incorrect number of brightness values provided for retina drawing. Expected {total_hexes}, got {len(brightness_values) if brightness_values else 0}. Drawing gray hexes.")
        brightness_values = [128] * total_hexes # Draw gray placeholders

    if canvas is None or canvas.size == 0:
        logger.error("Cannot draw retina on None or empty canvas.")
        return

    # Calculate bounding box width/height for centering calculations
    grid_bounding_width = 0
    grid_bounding_height = 0
    if grid_cols > 0 and grid_rows > 0 and hex_radius > 0:
        grid_bounding_width = hex_width * 0.75 * (grid_cols -1) + hex_width
        grid_bounding_height = hex_height * grid_rows + (hex_height / 2 if grid_cols > 1 else 0) # Accounts for staggering
        grid_bounding_width = max(1, int(round(grid_bounding_width)))
        grid_bounding_height = max(1, int(round(grid_bounding_height)))
    else:
        logger.error("Cannot calculate grid bounding box: invalid dimensions/radius.")
        return # Cannot proceed without bounding box


    # Calculate the origin (top-left corner) of the *entire grid area* relative to the provided canvas center
    grid_origin_x = center_x - grid_bounding_width / 2.0
    grid_origin_y = center_y - grid_bounding_height / 2.0

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate individual hexagon position relative to the grid origin
            offset_x = hex_width * 0.75 * col
            offset_y = hex_height * row
            if col % 2 != 0: # Odd columns are shifted down
                offset_y += hex_height / 2

            # Calculate absolute center of this hexagon on the canvas
            hex_center_x = grid_origin_x + offset_x + hex_width / 2.0 # Add half width to get to hex center from offset
            hex_center_y = grid_origin_y + offset_y + hex_height / 2.0 # Add half height

            vertices = get_hexagon_vertices(hex_center_x, hex_center_y, hex_radius)

            brightness = brightness_values[val_idx]
            # Ensure brightness is a valid number for color conversion
            if brightness is None or not np.isfinite(brightness):
                brightness = 0 # Default to black if invalid
            color_intensity = int(np.clip(brightness, 0, 255))
            hex_color = (color_intensity, color_intensity, color_intensity) # BGR for OpenCV drawing

            try:
                # Draw border first, then fill
                cv2.polylines(canvas, [vertices], isClosed=True, color=(50, 50, 50), thickness=1) # Dark gray border
                cv2.fillConvexPoly(canvas, vertices, color=hex_color)
            except Exception as e:
                 # Catch potential errors if vertices are somehow invalid (e.g., off-canvas)
                 logger.warning(f"Error drawing hexagon at grid ({row}, {col}), canvas pos ({hex_center_x:.1f}, {hex_center_y:.1f}): {e}")

            val_idx += 1
            # Safety break if somehow val_idx exceeds expected, though list length check handles this mostly
            if val_idx >= total_hexes: break
        if val_idx >= total_hexes: break


def simulate_nmf_retina_output(stitched_image, num_hex_rows, num_hex_cols, binocular_overlap_deg, total_fov_deg):
    """
    Simulates NMF retina input by sampling the stitched/concatenated image.
    Returns two lists of average brightness values for left and right simulated retinas.
    """
    num_hexes = num_hex_rows * num_hex_cols
    default_output = [0.0] * num_hexes # Use float for averages

    if stitched_image is None or stitched_image.ndim != 3 or stitched_image.shape[0] <= 0 or stitched_image.shape[1] <= 0:
        logger.warning("Cannot simulate retina output from None, non-3D, or zero-dimension image.")
        return default_output, default_output

    h, w, _ = stitched_image.shape

    # Avoid division by zero if total_fov_deg is invalid
    if total_fov_deg <= 0:
        logger.error(f"Cannot calculate pixels per degree with invalid total_fov_deg: {total_fov_deg}")
        return default_output, default_output
    pixels_per_degree = w / total_fov_deg

    overlap_pixels = int(round(binocular_overlap_deg * pixels_per_degree))
    # Ensure overlap isn't wider than the image itself
    overlap_pixels = min(w, max(0, overlap_pixels))

    # Calculate the width of each eye's FOV in pixels, ensuring it's valid
    eye_fov_pixels = (w + overlap_pixels) / 2.0
    eye_fov_pixels = min(w, max(0, eye_fov_pixels)) # Clamp between 0 and full width

    # Calculate pixel indices for slicing
    left_eye_end_px = int(round(eye_fov_pixels))
    right_eye_start_px = int(round(w - eye_fov_pixels))

    # Clamp indices to be within image bounds [0, w] for slicing
    left_eye_end_px = min(w, max(0, left_eye_end_px))
    right_eye_start_px = min(w, max(0, right_eye_start_px))

    # Handle edge case where calculations might lead to start >= end or minimal region
    if right_eye_start_px >= w: right_eye_start_px = max(0, w - 1)
    if left_eye_end_px <= 0 : left_eye_end_px = min(1, w) # Ensure at least 1 pixel if possible
    if right_eye_start_px >= left_eye_end_px and w > 0: # Check for non-sensical overlap/slicing
        logger.warning(f"Calculated eye regions have invalid overlap: LeftEnd={left_eye_end_px}, RightStart={right_eye_start_px}. Attempting split adjustment.")
        # Attempt a sensible default: split near middle with some overlap centered
        mid_point = w / 2.0
        half_overlap = overlap_pixels / 2.0
        left_eye_end_px = int(round(mid_point + half_overlap))
        right_eye_start_px = int(round(mid_point - half_overlap))
        # Re-clamp after adjustment
        left_eye_end_px = min(w, max(0, left_eye_end_px))
        right_eye_start_px = min(w, max(0, right_eye_start_px))
        # Ensure start < end after adjustment
        if right_eye_start_px >= left_eye_end_px:
             right_eye_start_px = max(0, left_eye_end_px - 1) # Force at least one pixel gap/overlap if possible

        logger.warning(f"Adjusted eye regions: LeftEnd={left_eye_end_px}, RightStart={right_eye_start_px}.")


    logger.debug(f"Img W:{w}px ({total_fov_deg:.1f}deg), Px/Deg:{pixels_per_degree:.2f}, Overlap:{overlap_pixels}px ({binocular_overlap_deg:.1f}deg)")
    logger.debug(f"Eye FOV:{eye_fov_pixels:.1f}px, Left Eye Slice:[0:{left_eye_end_px}], Right Eye Slice:[{right_eye_start_px}:{w}]")

    try:
        gray_stitched = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2GRAY)
    except cv2.error as e:
        logger.error(f"Failed to convert stitched image to grayscale: {e}")
        return default_output, default_output

    # Slice the image regions for each eye
    # Slicing [:, start:end] means end is exclusive
    left_eye_region = gray_stitched[:, 0:left_eye_end_px]
    right_eye_region = gray_stitched[:, right_eye_start_px:w]

    # Check if regions are valid before sampling
    if left_eye_region.size == 0: logger.warning(f"Left eye region is empty (slice 0:{left_eye_end_px}).")
    if right_eye_region.size == 0: logger.warning(f"Right eye region is empty (slice {right_eye_start_px}:{w}).")

    logger.debug(f"Left Eye Region Shape: {left_eye_region.shape}")
    logger.debug(f"Right Eye Region Shape: {right_eye_region.shape}")

    def sample_region(region, num_rows, num_cols):
        """ Samples average brightness from grid cells within the region. """
        num_target_hexes = num_rows * num_cols
        brightness_values = []

        if region is None or region.ndim != 2 or region.shape[0] <= 0 or region.shape[1] <= 0 or num_target_hexes == 0:
            logger.debug(f"Cannot sample region: Invalid input. Region shape: {region.shape if region is not None else 'None'}, Target Hexes: {num_target_hexes}")
            return [0.0] * num_target_hexes # Return default values

        h_reg, w_reg = region.shape

        # Avoid division by zero if rows/cols are zero
        row_step = h_reg / max(1, num_rows)
        col_step = w_reg / max(1, num_cols)

        # Avoid steps being zero if region dim is 1
        if row_step <= 0: row_step = 1.0
        if col_step <= 0: col_step = 1.0

        for r in range(num_rows):
            for c in range(num_cols):
                # Calculate start/end indices, using floor/ceil might be safer than round for coverage
                y_start = int(math.floor(r * row_step))
                y_end = int(math.ceil((r + 1) * row_step))
                x_start = int(math.floor(c * col_step))
                x_end = int(math.ceil((c + 1) * col_step))

                # Clamp indices to be within region bounds [0, H/W]
                y_start = min(h_reg, max(0, y_start))
                y_end = min(h_reg, max(0, y_end))
                x_start = min(w_reg, max(0, x_start))
                x_end = min(w_reg, max(0, x_end))

                # Ensure start < end for slicing, handle 1-pixel dimensions or collapsed regions
                if y_start >= y_end:
                    y_end = y_start + 1 # Ensure at least 1 pixel height if collapsed
                if x_start >= x_end:
                    x_end = x_start + 1 # Ensure at least 1 pixel width if collapsed

                # Re-clamp after potential adjustment
                y_end = min(h_reg, y_end)
                x_end = min(w_reg, x_end)

                # Extract patch, ensure slice is valid
                if y_start < y_end and x_start < x_end:
                    patch = region[y_start:y_end, x_start:x_end]
                else:
                    patch = None # Slice became invalid

                if patch is not None and patch.size > 0:
                    avg_brightness = np.mean(patch)
                    # Handle potential NaN/Inf from weird patches
                    if not np.isfinite(avg_brightness):
                        logger.warning(f"NaN/Inf brightness in patch [{y_start}:{y_end}, {x_start}:{x_end}], using 0.0")
                        avg_brightness = 0.0
                    brightness_values.append(float(avg_brightness))
                else:
                    # Log if patch is empty (should be less common with floor/ceil and adjustments)
                    logger.warning(f"Empty or invalid patch at R:{r}, C:{c} (Slice [{y_start}:{y_end}, {x_start}:{x_end}]) in region {h_reg}x{w_reg}. Appending 0.0")
                    brightness_values.append(0.0)

        # Pad if fewer values were generated than expected (safety net)
        if len(brightness_values) != num_target_hexes:
             logger.warning(f"Padding retina values (got {len(brightness_values)}, expected {num_target_hexes})")
             while len(brightness_values) < num_target_hexes:
                 brightness_values.append(0.0)

        return brightness_values[:num_target_hexes] # Ensure correct length

    left_brightness = sample_region(left_eye_region, num_hex_rows, num_hex_cols)
    right_brightness = sample_region(right_eye_region, num_hex_rows, num_hex_cols)

    return left_brightness, right_brightness


def create_visualizations(stitched_pov, left_retina_values, right_retina_values, scale_factor, hex_config):
    """ Creates the combined visualization with stitched view and hex grids. """
    placeholder_width = 320 # Estimate reasonable width for placeholder if needed
    if stitched_pov is None or stitched_pov.ndim != 3 or stitched_pov.shape[0] <= 0 or stitched_pov.shape[1] <= 0:
        logger.warning("create_visualizations received invalid stitched_pov. Creating placeholder.")
        # Use RGB format consistent with normal operation
        stitched_pov = np.zeros((64, placeholder_width, 3), dtype=np.uint8) + 50 # Gray placeholder (RGB)

    h_stitched, w_stitched, _ = stitched_pov.shape
    top_view = stitched_pov # Top part is the (potentially placeholder) stitched POV

    # Bottom canvas preparation
    bottom_canvas = None
    bottom_height = 0
    final_width = w_stitched # Start assuming final width is the stitched POV width

    if ENABLE_HEX_VISUALIZATION:
        hex_radius = hex_config['radius']
        grid_rows = hex_config['rows']
        grid_cols = hex_config['cols']

        # Calculate the dimensions needed to draw *one* hex grid
        grid_disp_width = 0
        grid_disp_height = 0
        if grid_cols > 0 and grid_rows > 0 and hex_radius > 0:
            hex_width = hex_radius * 2
            hex_height = math.sqrt(3) * hex_radius
            grid_disp_width = int(round(hex_width * 0.75 * (grid_cols - 1) + hex_width))
            grid_disp_height = int(round(hex_height * grid_rows + (hex_height / 2 if grid_cols > 1 else 0))) # Add stagger offset only if multiple cols
            grid_disp_width = max(1, grid_disp_width) # Ensure at least 1 pixel
            grid_disp_height = max(1, grid_disp_height)
        else:
             logger.warning("Hex grid dimensions/radius invalid, cannot calculate display size.")
             grid_disp_width = 100 # Default size if config is bad
             grid_disp_height = 100

        # Define padding and spacing for the bottom area
        bottom_padding = 20 # Padding around hex grids (top/bottom/left/right)
        bottom_spacing = 40 # Horizontal space between left and right grid

        # Calculate the minimum width needed for the two grids, spacing, and padding
        required_grid_canvas_width = (grid_disp_width * 2) + bottom_spacing + (bottom_padding * 2)

        # Determine the final width needed for the combined image
        final_width = max(w_stitched, required_grid_canvas_width)

        # Calculate bottom canvas height
        bottom_height = grid_disp_height + bottom_padding * 2

    else:
        # If hex vis disabled, minimal placeholder bar
        bottom_height = 20
        final_width = w_stitched # Match top view width

    # Resize top to fit
    top_view_resized = top_view
    if w_stitched != final_width:
        logger.debug(f"Resizing top view width from {w_stitched} to {final_width} to match bottom requirements.")
        # Calculate new height maintaining aspect ratio
        new_h_top = int(round(h_stitched * final_width / w_stitched))
        new_h_top = max(1, new_h_top) # Ensure height is at least 1
        try:
            top_view_resized = cv2.resize(top_view, (final_width, new_h_top), interpolation=cv2.INTER_NEAREST)
        except cv2.error as e:
            logger.error(f"Error resizing top view: {e}. Using original.")
            top_view_resized = top_view # Fallback
            final_width = top_view_resized.shape[1] # Reset final_width if resize failed
            logger.warning(f"Adjusted final_width back to {final_width} due to top view resize error.")
    # Get final dimensions of the (potentially resized) top view
    h_top_final, w_top_final, _ = top_view_resized.shape

    # Render bottom canvas
    # Ensure bottom canvas width *exactly* matches the final top view width
    bottom_canvas = np.zeros((bottom_height, w_top_final, 3), dtype=np.uint8) # Use w_top_final

    if ENABLE_HEX_VISUALIZATION:
        # Recalculate drawing centers based on the *actual* bottom_canvas dimensions (w_top_final, bottom_height)
        # Center Y is simply the middle of the bottom canvas height
        grid_draw_center_y = bottom_height / 2.0

        # Calculate available horizontal space for drawing grids within padding
        available_draw_width = w_top_final - 2 * bottom_padding
        if available_draw_width < (grid_disp_width * 2 + bottom_spacing):
             logger.warning(f"Canvas width ({w_top_final}) might not be enough for grids ({grid_disp_width}x2), spacing ({bottom_spacing}), and padding ({bottom_padding}x2). Grids might overlap or touch padding.")
             # Adjust spacing to fit if needed? Or let it overlap? For now, proceed with calculated centers.
             # Alternative: scale down grids if too wide? Complex.

        # Center X for left grid: place it so its left edge is at `bottom_padding`
        left_grid_draw_center_x = bottom_padding + grid_disp_width / 2.0
        # Center X for right grid: place it so its right edge is at `w_top_final - bottom_padding`
        right_grid_draw_center_x = w_top_final - bottom_padding - grid_disp_width / 2.0

        # Draw the grids and labels using the final calculated centers
        draw_simulated_retina(bottom_canvas, left_grid_draw_center_x, grid_draw_center_y, grid_rows, grid_cols, hex_radius, left_retina_values)
        # Position text relative to the calculated grid origin
        left_text_x = int(round(left_grid_draw_center_x - grid_disp_width / 2.0))
        left_text_x = max(0, left_text_x) # Ensure text doesn't start off-screen left
        cv2.putText(bottom_canvas, "Left Retina (Sim)",
                    (left_text_x, bottom_padding - 5), # Text position slightly above grid origin
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        draw_simulated_retina(bottom_canvas, right_grid_draw_center_x, grid_draw_center_y, grid_rows, grid_cols, hex_radius, right_retina_values)
        # Position text relative to the calculated grid origin
        right_text_x = int(round(right_grid_draw_center_x - grid_disp_width / 2.0))
        right_text_x = max(0, right_text_x) # Ensure text doesn't start off-screen left
        cv2.putText(bottom_canvas, "Right Retina (Sim)",
                    (right_text_x, bottom_padding - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    else:
        # Draw placeholder text if hex vis disabled
        cv2.putText(bottom_canvas, "Retina vis Disabled", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)


    # Stitch top and bottom canvases
    final_canvas = None
    try:
        # Verify widths one last time before stacking
        if top_view_resized.shape[1] != bottom_canvas.shape[1]:
             logger.error(f"Width mismatch before vstack! Top: {top_view_resized.shape[1]}, Bottom: {bottom_canvas.shape[1]}. Attempting resize fallback.")
             # Attempt to resize bottom to match top as a last resort
             bottom_canvas = cv2.resize(bottom_canvas, (top_view_resized.shape[1], bottom_canvas.shape[0]), interpolation=cv2.INTER_NEAREST)

        final_canvas = np.vstack((top_view_resized, bottom_canvas))
    except Exception as e:
        logger.error(f"Error during vstack: {e}. Top shape: {top_view_resized.shape}, Bottom shape: {bottom_canvas.shape}", exc_info=True)
        # Fallback: return just the resized top view if stacking fails
        final_canvas = top_view_resized

    if final_canvas is None:
        logger.error("Final canvas is None after stacking attempt.")
        return None # Cannot proceed

    # Scale
    final_canvas_scaled = final_canvas # Default to unscaled
    final_h, final_w, _ = final_canvas.shape

    if final_w > 0 and final_h > 0 and scale_factor != 1.0 and scale_factor > 0:
        target_w = int(round(final_w * scale_factor))
        target_h = int(round(final_h * scale_factor))
        target_w = max(1, target_w) # Ensure target dimensions are positive
        target_h = max(1, target_h)
        try:
            final_canvas_scaled = cv2.resize(final_canvas, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        except cv2.error as e:
            logger.error(f"Error scaling final canvas: {e}. Returning unscaled.")
            final_canvas_scaled = final_canvas # Fallback to unscaled
    elif scale_factor <= 0:
         logger.warning(f"Invalid scale_factor ({scale_factor}), returning unscaled image.")

    return final_canvas_scaled


def get_obstacle_avoidance_action(env, left_vals, right_vals, threshold, turn_magnitude):
    """ Basic obstacle avoidance: turn away from dark areas. Returns full action dict."""
    action = env.action_space.noop() # Start with a blank action

    # Handle cases with missing or invalid retina data
    if not left_vals or not right_vals:
        logger.debug("Avoidance: No valid retina data, defaulting to forward.")
        if ALWAYS_MOVE_FORWARD: action['forward'] = 1
        if ALWAYS_JUMP: action['jump'] = 1
        if ALWAYS_SPRINT: action['sprint'] = 1
        action['camera'] = np.array([0, 0.0], dtype=np.float32) # Ensure camera key exists
        return action

    # Calculate average brightness, ignoring potential NaN/Inf/None values
    finite_left = [v for v in left_vals if v is not None and np.isfinite(v)]
    finite_right = [v for v in right_vals if v is not None and np.isfinite(v)]

    # Use 128 (mid-gray) as default if no valid finite values found
    avg_left = np.mean(finite_left) if finite_left else 128.0
    avg_right = np.mean(finite_right) if finite_right else 128.0

    logger.debug(f"Avg Brightness - Left: {avg_left:.1f}, Right: {avg_right:.1f} (Threshold: {threshold})")

    turn = 0.0
    move_forward = False # Decide whether to activate forward motion

    # Determine if areas are considered 'dark' based on the threshold
    left_dark = avg_left < threshold
    right_dark = avg_right < threshold

    # Simple avoidance logic
    if left_dark and right_dark:
        # Both sides dark: Obstacle likely straight ahead. Turn consistently (e.g., right)
        logger.debug("Avoidance: Both sides dark, turning right.")
        turn = turn_magnitude # Turn right
        move_forward = True # Still try to move forward while turning (or maybe stop?)
    elif left_dark:
        # Left is dark, turn right (positive yaw)
        logger.debug("Avoidance: Left side dark, turning right.")
        turn = turn_magnitude
        move_forward = True
    elif right_dark:
        # Right is dark, turn left (negative yaw)
        logger.debug("Avoidance: Right side dark, turning left.")
        turn = -turn_magnitude
        move_forward = True
    else:
        # Path seems clear, go straight
        logger.debug("Avoidance: Path clear, moving forward.")
        turn = 0.0
        move_forward = True

    # Populate the action dictionary based on decisions and global flags
    action['forward'] = 1 if move_forward and ALWAYS_MOVE_FORWARD else 0
    action['jump'] = 1 if move_forward and ALWAYS_JUMP else 0 # Only jump if moving forward
    action['sprint'] = 1 if move_forward and ALWAYS_SPRINT else 0 # Only sprint if moving forward

    # Set the camera action (pitch is 0, yaw is the calculated turn)
    action['camera'] = np.array([0, turn], dtype=np.float32)

    return action


if __name__ == "__main__":
    env = None
    video_writer = None
    done = False
    max_steps = 1000 # Limit the number of agent steps

    # Set logger level for main execution (DEBUG for details, INFO for cleaner output)
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    try:
        logger.info("Creating MineRL environment (MineRLNavigateDense-v0)...")
        # Note: MineRLNavigateDense requires compass use. For pure visual navigation,
        # MineRLNavigate-v0 might suffice, but Dense provides denser rewards.
        env = gym.make('MineRLNavigateDense-v0')
        logger.info("Environment created successfully.")
    except gym.error.UnregisteredEnv:
        logger.error("MineRLNavigateDense-v0 not found. Ensure MineRL v0.4.4 is installed and registered.", exc_info=True)
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error creating environment: {e}", exc_info=True)
        exit(1)

    try:
        logger.info("Resetting environment...")
        initial_obs = env.reset()
        if initial_obs is None:
            raise ValueError("Initial env.reset() returned None observation.")
        # Store the current observation in the environment object for helpers
        env.current_obs = initial_obs
        logger.info("Environment reset successfully.")
    except Exception as e:
        logger.error(f"Error during initial env.reset(): {e}", exc_info=True)
        if env: env.close()
        exit(1)

    # Config for hex grid visualization
    hex_config = {'rows': HEX_GRID_ROWS, 'cols': HEX_GRID_COLS, 'radius': HEX_GRID_RADIUS}

    # Create display window
    cv2.namedWindow("NMF Agent View (Minecraft)", cv2.WINDOW_NORMAL) # Allow resizing

    try:
        # Use tqdm context manager for progress bar linked to logging
        with tqdm(total=max_steps, desc="Agent Steps", unit="step", leave=True) as pbar:
            for step_counter in range(max_steps):
                if done:
                    logger.info(f"Episode finished, breaking loop at step {step_counter}.")
                    break # Exit the main loop

                step_start_time = time.time()
                logger.debug(f"--- Step {step_counter+1} ---")

                # Sim widened stitched FOV
                # This function performs multiple env.steps internally and resets camera.
                stitched_pov, _, fov_done = simulate_wide_fov_mineRL(
                    env, num_captures, yaw_per_capture_step, initial_yaw_turn
                )
                done = done or fov_done # Update done flag if FOV simulation ended episode

                # Prepare data for visualization even if done, to show final frame
                left_retina_brightness = []
                right_retina_brightness = []
                if stitched_pov is not None:
                    # Call NMF sim
                    left_retina_brightness, right_retina_brightness = simulate_nmf_retina_output(
                        stitched_pov, HEX_GRID_ROWS, HEX_GRID_COLS, NMF_BINOCULAR_OVERLAP_DEG, TARGET_FOV
                    )
                else:
                    # Use default values if POV is None (e.g., if fov_done is True)
                    logger.warning("Stitched POV is None, using default brightness for visualization.")
                    num_hexes = HEX_GRID_ROWS * HEX_GRID_COLS
                    left_retina_brightness = [0.0] * num_hexes
                    right_retina_brightness = [0.0] * num_hexes


                # Create visualization frame (using potentially None POV or default retina values)
                display_image = create_visualizations(
                    stitched_pov, left_retina_brightness, right_retina_brightness, DISPLAY_SCALE_FACTOR, hex_config
                )

                # Show + save vid frame
                if display_image is not None and display_image.shape[0] > 0 and display_image.shape[1] > 0:
                    # Convert final image to BGR for OpenCV display/saving
                    display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("NMF Agent View (Minecraft)", display_image_bgr)

                    # Initialize VideoWriter on the first valid frame
                    if SAVE_VIDEO and video_writer is None:
                        try:
                            h_vid, w_vid = display_image_bgr.shape[:2] # Use BGR frame dimensions
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
                            video_writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, OUTPUT_FPS, (w_vid, h_vid))
                            if video_writer.isOpened():
                                logger.info(f"Video writer initialized. Saving to {OUTPUT_FILENAME} ({w_vid}x{h_vid}) at {OUTPUT_FPS} FPS.")
                            else:
                                logger.error(f"Could not open video writer for {OUTPUT_FILENAME}.")
                                video_writer = None # Ensure it stays None if failed
                        except Exception as e:
                            logger.error(f"Error initializing video writer: {e}", exc_info=True)
                            video_writer = None
                            SAVE_VIDEO = False # Disable saving if init fails

                    # Write frame if writer is ready
                    if SAVE_VIDEO and video_writer is not None and video_writer.isOpened():
                        try:
                           video_writer.write(display_image_bgr)
                        except Exception as e:
                            logger.error(f"Error writing frame {step_counter+1} to video: {e}", exc_info=True)
                            # Continue running, but log the error

                else:
                    logger.warning("create_visualizations returned invalid image. Cannot display or write.")


                # Action selection and env step
                # If episode ended during FOV sim/vis, skip action/step
                if done:
                    logger.info(f"Episode ended at step {step_counter+1} (likely during FOV sim/reset). Skipping action/step.")
                    pbar.update(1)
                    cv2.waitKey(1) # Allow window update
                    continue # Go to next loop iteration (will break immediately)

                # If stitched_pov was None earlier, we need to decide action (e.g., no-op)
                if stitched_pov is None:
                    logger.warning("Failed to get valid POV for this step. Agent performing no-op action.")
                    combined_action = env.action_space.noop()
                else:
                    # Get action based on retina input
                    combined_action = get_obstacle_avoidance_action(
                        env, left_retina_brightness, right_retina_brightness, BRIGHTNESS_THRESHOLD, TURN_MAGNITUDE
                    )
                logger.debug(f"Calculated Action: {combined_action}")

                # Run action
                try:
                    obs, reward, step_done, info = env.step(combined_action)
                    done = step_done # Update done status based on this step
                    if obs is not None:
                        env.current_obs = obs # Update observation
                    else:
                        logger.warning("env.step() returned None observation after combined action.")
                        done = True # Treat as episode end

                    # Log reward if desired
                    if reward != 0: logger.debug(f"Step {step_counter+1}: Reward = {reward}")

                except Exception as e:
                    logger.error(f"Error during main agent env.step(): {e}", exc_info=True)
                    done = True # Assume episode ends on error


                # Loop cleanup
                step_end_time = time.time()
                logger.debug(f"Step {step_counter+1} processing took {step_end_time - step_start_time:.3f} seconds.")

                pbar.update(1) # Update progress bar

                # Check for quit key ('q') pressed in the OpenCV window
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User pressed 'q', exiting loop.")
                    break
                # Add other keybinds if needed, e.g., pause

            # Log if loop finished due to max_steps rather than 'done' flag
            if step_counter == max_steps - 1 and not done :
                logger.info(f"Reached maximum steps ({max_steps}) without episode finishing.")

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected, shutting down.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        # Cleanup resources
        logger.info("Cleaning up resources...")
        cv2.destroyAllWindows()
        logger.info("OpenCV windows destroyed.")

        if video_writer is not None and video_writer.isOpened():
            logger.info(f"Releasing video writer for {OUTPUT_FILENAME}...")
            video_writer.release()
            logger.info("Video writer released.")
        elif SAVE_VIDEO:
            logger.info("Video writer was not initialized or already closed.")

        if env:
            try:
                env.close()
                logger.info("MineRL environment closed.")
            except Exception as e:
                logger.error(f"Error closing environment: {e}", exc_info=True)

        logger.info("Cleanup complete.")
        print("Script finished.")