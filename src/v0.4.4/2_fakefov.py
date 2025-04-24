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
            self.flush() # Not strictly necessary for tqdm but good practice
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
ENABLE_HEX_VISUALIZATION = True # Set to False to disable retina viz
HEX_GRID_RADIUS = 3 # Controls size of hexagons in visualization
HEX_GRID_COLS = 12 # Approximate number of columns for simulated retina hex grid
HEX_GRID_ROWS = 10 # Approximate number of rows
# Adjusted spacing between grids to give labels more room
BOTTOM_GRID_SPACING = 60 # Horizontal space between left and right grid

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
logger.info(f"Bottom Grid Spacing: {BOTTOM_GRID_SPACING} px")
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
        # Use the current observation if available, otherwise use the last known one
        current_obs = getattr(env, 'current_obs', None)
        if current_obs is None:
             logger.warning("turn_agent called without a valid env.current_obs.")
             # Optionally, perform a no-op step to try and get an observation
             # obs, _, done, _ = env.step(env.action_space.noop())
             # if obs is not None: env.current_obs = obs
             # else: return None, True # Failed to get obs
             return None, True # Cannot proceed without an observation state

        obs, _, done, _ = env.step(action)

        if obs is not None:
            env.current_obs = obs # Update shared observation state
        else:
            logger.warning("turn_agent received None observation from env.step. Episode might have ended.")
            # Keep previous obs if current is None, let done flag handle termination
            done = True # Assume done if observation is None after an action
        return env.current_obs, done # Return the potentially updated current_obs
    except Exception as e:
        logger.error(f"Error during env.step() in turn_agent: {e}", exc_info=True)
        return getattr(env, 'current_obs', None), True # Return last known obs, assume done


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
            frames.append(current_pov.copy()) # Append a copy
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
            stitched_image_rgb = env.current_obs['pov'].copy()
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

    # Calculate approximate bounding box for centering calculation within this function
    grid_bounding_width = hex_width * 0.75 * (grid_cols - 1) + hex_width if grid_cols > 0 else 0
    grid_bounding_height = hex_height * (grid_rows + 0.5) if grid_cols > 1 else hex_height # Accounts for staggering only if needed

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate position based on grid layout (staggered columns)
            offset_x = hex_width * 0.75 * col
            offset_y = hex_height * row
            if col % 2 != 0: # Odd columns are shifted down
                offset_y += hex_height / 2

            # Calculate absolute center of this hexagon
            # Top-left corner of the bounding box for the grid area relative to the provided center_x, center_y:
            grid_origin_x = center_x - grid_bounding_width / 2
            grid_origin_y = center_y - grid_bounding_height / 2

            hex_center_x = grid_origin_x + offset_x + hex_width / 2 # Add half width to get to hex center from offset
            hex_center_y = grid_origin_y + offset_y + hex_height / 2 # Add half height

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
    # Symmetrically: Eye FOV = (Total Width + Overlap) / 2
    eye_fov_pixels = (w + overlap_pixels) / 2.0
    eye_fov_pixels = min(w, max(0, eye_fov_pixels)) # Clamp between 0 and full width

    # Calculate pixel indices for slicing
    # Left eye goes from pixel 0 to eye_fov_pixels
    left_eye_end_px = int(round(eye_fov_pixels))
    # Right eye goes from (w - eye_fov_pixels) to w
    right_eye_start_px = int(round(w - eye_fov_pixels))

    # Clamp indices to be within image bounds [0, w] for slicing
    left_eye_end_px = min(w, max(0, left_eye_end_px))
    right_eye_start_px = min(w, max(0, right_eye_start_px))

    # Handle edge case where calculations might lead to start >= end
    if right_eye_start_px >= w: right_eye_start_px = max(0, w - 1)
    if left_eye_end_px <= 0 : left_eye_end_px = min(1, w) # Ensure at least 1 pixel if possible
    if right_eye_start_px >= left_eye_end_px and w > 0: # Check for non-sensical overlap/slicing
        logger.warning(f"Calculated eye regions have invalid overlap: LeftEnd={left_eye_end_px}, RightStart={right_eye_start_px}. Adjusting.")
        # Attempt a sensible default: split in middle with overlap centered
        mid_point = w // 2
        left_eye_end_px = mid_point + overlap_pixels // 2
        right_eye_start_px = mid_point - overlap_pixels // 2
        left_eye_end_px = min(w, max(0, left_eye_end_px))
        right_eye_start_px = min(w, max(0, right_eye_start_px))
        # Ensure start is still less than end after adjustment
        if right_eye_start_px >= left_eye_end_px:
             right_eye_start_px = max(0, left_eye_end_px -1) # Force at least 1 pixel overlap if possible
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

        # Avoid steps being zero if region dim is 1 or less
        if row_step <= 1e-6 : row_step = h_reg # Treat as single row if height too small
        if col_step <= 1e-6 : col_step = w_reg # Treat as single col if width too small

        for r in range(num_rows):
            for c in range(num_cols):
                # Calculate start/end indices, rounding needed
                y_start = int(round(r * row_step))
                y_end = int(round((r + 1) * row_step))
                x_start = int(round(c * col_step))
                x_end = int(round((c + 1) * col_step))

                # Clamp indices to be within region bounds [0, H/W]
                y_start = min(h_reg, max(0, y_start))
                y_end = min(h_reg, max(0, y_end))
                x_start = min(w_reg, max(0, x_start))
                x_end = min(w_reg, max(0, x_end))

                # Ensure start < end for slicing, handle 1-pixel or zero-dim slices
                if y_start >= y_end:
                     if y_start > 0: y_start = y_end - 1
                     else: y_end = y_start + 1
                     y_start = max(0, y_start) # Re-clamp
                     y_end = min(h_reg, y_end)

                if x_start >= x_end:
                     if x_start > 0: x_start = x_end - 1
                     else: x_end = x_start + 1
                     x_start = max(0, x_start)
                     x_end = min(w_reg, x_end)

                # Extract patch only if slice dimensions are valid
                if y_start < y_end and x_start < x_end:
                     patch = region[y_start:y_end, x_start:x_end]
                     if patch.size > 0:
                         avg_brightness = np.mean(patch)
                         # Handle potential NaN/Inf
                         if not np.isfinite(avg_brightness):
                             logger.warning(f"NaN/Inf brightness in patch [{y_start}:{y_end}, {x_start}:{x_end}], using 0.0")
                             avg_brightness = 0.0
                         brightness_values.append(float(avg_brightness))
                     else:
                         # Slice was valid but patch is empty (should be rare)
                         logger.warning(f"Empty patch despite valid slice at R:{r}, C:{c} (Slice [{y_start}:{y_end}, {x_start}:{x_end}]) in region {h_reg}x{w_reg}. Appending 0.0")
                         brightness_values.append(0.0)
                else:
                     # Slice dimensions were invalid after clamping/adjustment
                     logger.warning(f"Invalid slice dimensions at R:{r}, C:{c} (Slice [{y_start}:{y_end}, {x_start}:{x_end}]) in region {h_reg}x{w_reg}. Appending 0.0")
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
        # Make placeholder height similar to expected MineRL height for better visualization
        placeholder_height = 64
        stitched_pov = np.zeros((placeholder_height, placeholder_width, 3), dtype=np.uint8) + 50 # Gray placeholder

    h_stitched, w_stitched, _ = stitched_pov.shape
    top_view = stitched_pov # Top part is the (potentially placeholder) stitched POV

    # Bottom canvas
    if ENABLE_HEX_VISUALIZATION:
        hex_radius = hex_config['radius']
        grid_rows = hex_config['rows']
        grid_cols = hex_config['cols']
        bottom_spacing = hex_config['spacing'] # Get spacing from config

        # Calculate the dimensions needed to draw *one* hex grid
        if grid_cols > 0 and grid_rows > 0 and hex_radius > 0:
            hex_width = hex_radius * 2
            hex_height = math.sqrt(3) * hex_radius
            # Approx width: 0.75 width per col except last one + full last width
            grid_disp_width = int(round(hex_width * 0.75 * (grid_cols - 1) + hex_width)) if grid_cols > 1 else int(round(hex_width))
            # Approx height: accounts for row height + half height for staggering
            grid_disp_height = int(round(hex_height * grid_rows + (hex_height / 2 if grid_cols > 1 else 0))) if grid_rows > 0 else 0
            grid_disp_width = max(1, grid_disp_width) # Ensure at least 1 pixel
            grid_disp_height = max(1, grid_disp_height)
        else:
            logger.warning("Hex grid dimensions/radius invalid, cannot calculate display size.")
            grid_disp_width = 100 # Default size if config is bad
            grid_disp_height = 100

        # Define padding for the bottom area
        bottom_padding = 20 # Min padding around hex grids (top/bottom/left/right)

        # Calculate the width needed for the two grids area (including space between them)
        total_grid_area_width = (grid_disp_width * 2) + bottom_spacing

        # Bottom canvas width must accommodate grids+padding OR match top view width, whichever is larger
        required_grid_canvas_width = total_grid_area_width + (bottom_padding * 2) # Min width needed for grids+padding
        bottom_width = max(w_stitched, required_grid_canvas_width)

        # Bottom canvas height needs grid height plus top/bottom padding
        # Add extra space for labels at the top
        label_area_height = 20 # Approximate height needed for labels
        bottom_height = grid_disp_height + bottom_padding * 2 + label_area_height

        # Create the black background canvas for the bottom part
        bottom_canvas = np.zeros((bottom_height, bottom_width, 3), dtype=np.uint8)

        # Calculate drawing centers for the grids to ensure they fit and are centered
        # Center Y, accounting for label area height
        grid_draw_center_y = label_area_height + bottom_padding + grid_disp_height / 2.0

        # Calculate margins to center the total grid area within the bottom_width
        leftover_space = bottom_width - total_grid_area_width
        side_margin = max(bottom_padding, leftover_space / 2.0) # Ensure at least min padding

        # Calculate grid centers based on the margins
        left_grid_draw_center_x = side_margin + grid_disp_width / 2.0
        right_grid_draw_center_x = bottom_width - side_margin - grid_disp_width / 2.0

        # Ensure centers are floats for precision
        left_grid_draw_center_x = float(left_grid_draw_center_x)
        right_grid_draw_center_x = float(right_grid_draw_center_x)
        grid_draw_center_y = float(grid_draw_center_y)

        # Draw the grids
        draw_simulated_retina(bottom_canvas, left_grid_draw_center_x, grid_draw_center_y, grid_rows, grid_cols, hex_radius, left_retina_values)
        draw_simulated_retina(bottom_canvas, right_grid_draw_center_x, grid_draw_center_y, grid_rows, grid_cols, hex_radius, right_retina_values)

        # Center labels
        left_text = "Left Retina (Sim)"
        right_text = "Right Retina (Sim)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (200, 200, 200)
        text_y_pos = bottom_padding # Place text baseline at the top padding line

        # Calculate text sizes
        (left_text_width, left_text_height), _ = cv2.getTextSize(left_text, font, font_scale, font_thickness)
        (right_text_width, right_text_height), _ = cv2.getTextSize(right_text, font, font_scale, font_thickness)

        # Calculate centered X positions for text
        left_text_x = int(round(left_grid_draw_center_x - left_text_width / 2.0))
        right_text_x = int(round(right_grid_draw_center_x - right_text_width / 2.0))

        # Ensure text starts within canvas bounds
        left_text_x = max(0, left_text_x)
        right_text_x = max(0, right_text_x)
        text_y_pos = max(left_text_height + 5, text_y_pos) # Ensure Y is below top edge

        # Draw centered text
        cv2.putText(bottom_canvas, left_text, (left_text_x, text_y_pos), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(bottom_canvas, right_text, (right_text_x, text_y_pos), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    else:
        # If hex vis disabled, create a small placeholder bar at the bottom
        bottom_height = 20
        bottom_width = w_stitched # Match top view width
        bottom_canvas = np.zeros((bottom_height, bottom_width, 3), dtype=np.uint8)
        cv2.putText(bottom_canvas, "Retina Viz Disabled", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # Stitch top and bottom
    # Resize top view (stitched POV) to match the final width of the bottom canvas if needed
    # Use INTER_NEAREST for pixelated look, avoid blur on upscale/downscale
    if bottom_width != w_stitched:
        logger.debug(f"Resizing top view from {w_stitched} to {bottom_width} width.")
        # Calculate new height maintaining aspect ratio
        new_h_top = int(round(h_stitched * bottom_width / w_stitched)) if w_stitched > 0 else h_stitched
        new_h_top = max(1, new_h_top) # Ensure height is at least 1
        try:
            # Ensure target width is also positive
            target_w_top = max(1, bottom_width)
            top_view_resized = cv2.resize(top_view, (target_w_top, new_h_top), interpolation=cv2.INTER_NEAREST)
        except cv2.error as e:
            logger.error(f"Error resizing top view: {e}. Using original.")
            top_view_resized = top_view # Fallback
            # If fallback, sizes won't match - resize bottom canvas *back* to match top
            bottom_width = top_view_resized.shape[1]
            try:
                 bottom_canvas = cv2.resize(bottom_canvas, (bottom_width, bottom_height), interpolation=cv2.INTER_NEAREST)
                 logger.warning(f"Adjusted bottom canvas width back to {bottom_width} due to top view resize error.")
            except cv2.error as e_bottom:
                 logger.error(f"Error resizing bottom canvas during fallback: {e_bottom}. Visualization may be misaligned.")
                 # If bottom resize fails too, we might have differently sized images to stack.
                 # Safest fallback: create empty final canvas? Or use just top view?
                 final_canvas = top_view_resized # Default to top view only if bottom resize fails
                 logger.error("Using only top view for visualization due to resize errors.")


    else:
        top_view_resized = top_view # No resize needed

    # Check if final_canvas was already assigned due to error
    if 'final_canvas' not in locals():
        h_top_resized, w_top_resized, _ = top_view_resized.shape
        h_bottom, w_bottom, _ = bottom_canvas.shape

        # Ensure widths match before stacking, padding if necessary
        if w_top_resized != w_bottom:
            logger.warning(f"Width mismatch before vstack: top={w_top_resized}, bottom={w_bottom}. Adjusting bottom.")
            # Resize bottom to match top as a final attempt
            try:
                 bottom_canvas = cv2.resize(bottom_canvas, (w_top_resized, h_bottom), interpolation=cv2.INTER_NEAREST)
                 w_bottom = bottom_canvas.shape[1] # update width
            except cv2.error as e_final_resize:
                 logger.error(f"Final resize attempt for bottom canvas failed: {e_final_resize}. Stacking may fail.")
                 # If resizing fails, cannot stack. Use top view only.
                 final_canvas = top_view_resized
                 logger.error("Using only top view for visualization due to final resize error.")


        # Create the final canvas by stacking vertically if widths match (and not already assigned)
        if 'final_canvas' not in locals() and w_top_resized == w_bottom:
             final_canvas = np.vstack((top_view_resized, bottom_canvas))
        elif 'final_canvas' not in locals():
             # Should not happen if resize fallback worked, but as safety:
             logger.error("Cannot vstack images of different widths. Returning only top view.")
             final_canvas = top_view_resized


    # Scale final output for easier display
    final_canvas_scaled = None
    final_h, final_w = final_canvas.shape[:2] # Get dimensions from the actual final canvas

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
    else:
        final_canvas_scaled = final_canvas # No scaling needed or invalid scale factor

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
        move_forward = True # Still try to move forward while turning (can be adjusted)
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
    # Only activate movement if decided above AND corresponding global flag is True
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

    # Config for hex grid visualization passed to the function
    hex_config = {
        'rows': HEX_GRID_ROWS,
        'cols': HEX_GRID_COLS,
        'radius': HEX_GRID_RADIUS,
        'spacing': BOTTOM_GRID_SPACING # Include spacing here
    }


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

                # Attempt final visualization even if episode ended during FOV sim
                final_visualization_attempted = False
                if done and not final_visualization_attempted:
                     logger.info(f"Episode ended during FOV simulation/reset at step {step_counter+1}. Attempting final viz.")
                     final_visualization_attempted = True # Prevent multiple attempts if already done
                     if stitched_pov is not None:
                          try:
                              left_b, right_b = simulate_nmf_retina_output(stitched_pov, HEX_GRID_ROWS, HEX_GRID_COLS, NMF_BINOCULAR_OVERLAP_DEG, TARGET_FOV)
                              display_img = create_visualizations(stitched_pov, left_b, right_b, DISPLAY_SCALE_FACTOR, hex_config)
                              if display_img is not None and display_img.shape[0] > 0 and display_img.shape[1] > 0:
                                  display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                                  cv2.imshow("NMF Agent View (Minecraft)", display_img_bgr)
                                  if SAVE_VIDEO and video_writer is not None and video_writer.isOpened():
                                      logger.debug("Writing final frame to video after episode end.")
                                      video_writer.write(display_img_bgr)
                                  cv2.waitKey(50) # Short delay
                          except Exception as viz_err:
                              logger.error(f"Error during final visualization after episode end: {viz_err}")
                     pbar.update(1) # Update progress bar for this step
                     continue # Skip rest of loop, will break on next iteration's 'if done'


                # Check if FOV simulation failed to produce an image
                if stitched_pov is None:
                    logger.warning("Failed to get valid POV for this step. Agent performing no-op.")
                    action = env.action_space.noop()
                    try:
                        if not done: # Don't step if already done
                            obs, _, step_done, _ = env.step(action)
                            done = done or step_done
                            if obs is not None: env.current_obs = obs
                            else:
                                logger.error("No-op step returned None observation after failed FOV.")
                                done = True
                    except Exception as e:
                        logger.error(f"Error during no-op step after failed POV: {e}", exc_info=True)
                        done = True
                    pbar.update(1) # Count this as a step
                    continue # Skip to next step

                # Call NMF sim
                left_retina_brightness, right_retina_brightness = simulate_nmf_retina_output(
                    stitched_pov, HEX_GRID_ROWS, HEX_GRID_COLS, NMF_BINOCULAR_OVERLAP_DEG, TARGET_FOV
                )

                # Create visualization frame
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
                                SAVE_VIDEO = False # Disable saving if cannot open
                        except Exception as e:
                            logger.error(f"Error initializing video writer: {e}", exc_info=True)
                            video_writer = None
                            SAVE_VIDEO = False # Disable saving if init fails

                    # Write frame if writer is ready
                    if SAVE_VIDEO and video_writer is not None and video_writer.isOpened():
                        video_writer.write(display_image_bgr)

                else:
                    logger.warning("create_visualizations returned invalid image. Cannot display or write.")

                # Get action based on retina input
                combined_action = get_obstacle_avoidance_action(
                    env, left_retina_brightness, right_retina_brightness, BRIGHTNESS_THRESHOLD, TURN_MAGNITUDE
                )
                logger.debug(f"Calculated Action: {combined_action}")

                # Run action
                try:
                    if not done: # Only step if the episode isn't already marked as done
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