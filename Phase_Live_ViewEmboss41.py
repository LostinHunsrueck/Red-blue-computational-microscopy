# ============================
# FULL PHASE CONTRAST LIVE VIEW
# with v-mode intensity cutoff
# ============================

import cv2
import numpy as np
from datetime import datetime
import os
import sys

# ============================================================
#  PATH HANDLING
# ============================================================

if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

os.chdir(script_dir)

# ============================================================
#  CAMERA INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 25)

if not cap.isOpened():
    print("ERROR: Could not open camera on port 0")
    exit(1)

# ============================================================
#  PARAMETERS
# ============================================================

PHASECORR_SIZE = (256, 256)
ALPHA = 0.2
MAX_SHIFT_PER_FRAME = 0.5
USE_CLAHE = True

use_noise_reduction = False
show_scale_bar = False
white_background = False

contrast_mode = 0
p_mode = 0
emboss_mode = 0
improve_mode = 0

# NEW: intensity cutoff mode
v_mode = 0

avg_buffer = []
AVG_SIZE = 3

# ============================================================
#  STARTUP HELP
# ============================================================

print("=== CONTROLS ===")
print("ESC  - Quit")
print("p    - Cycle view (phase / overlay / raw)")
print("n    - Toggle noise reduction")
print("m    - Toggle scale bar")
print("c    - Cycle contrast modes")
print("w    - Toggle soft white background")
print("e    - Cycle emboss modes")
print("i    - Cycle improvement (off / micro / medium / strong)")
print("v    - Cycle intensity cutoff (off / 20% / 40%)")
print("s    - Save normal frame (PNG)")
print("q    - Super-resolution still (3-frame average, PNG)")
print("1    - Start/stop recording (25 FPS WMV)")
print("2    - Start/stop recording (time-lapse 3 FPS → 25 FPS)")
print("3    - Start/stop recording (time-lapse 10 FPS → 25 FPS)")
print("================")

# ============================================================
#  HELPERS
# ============================================================

def phase_correlation_shift(img_ref, img_mov):
    ref = img_ref.astype(np.float32)
    mov = img_mov.astype(np.float32)
    shift, _ = cv2.phaseCorrelate(ref, mov)
    dy, dx = shift
    return dx, dy

def apply_subpixel_shift(img, dx, dy):
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

def create_center_roi(img, fraction=0.5):
    h, w = img.shape
    cw, ch = int(w * fraction), int(h * fraction)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return x0, y0, cw, ch

def draw_scale_bar(img, bar_px, bar_um):
    vis = img.copy()
    h, w = vis.shape[:2]
    margin = 20
    x1 = margin
    y1 = h - margin
    x2 = x1 + bar_px
    cv2.line(vis, (x1, y1), (x2, y1), (255, 255, 255), 2)
    cv2.putText(vis, f"{bar_um:.0f} um", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis

# ============================================================
#  CONTRAST MODES
# ============================================================

def process_clean(img):
    return img

def process_medium(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_local = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe_local.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0,0), 1.0)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)

def process_extreme(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_local = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4,4))
    l2 = clahe_local.apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0,0), 1.2)
    out = cv2.addWeighted(out, 2.2, blur, -1.2, 0)
    b, g, r = cv2.split(out)
    M = np.float32([[1, 0, 0.4], [0, 1, 0.4]])
    b_shift = cv2.warpAffine(b, M, (b.shape[1], b.shape[0]))
    return cv2.merge((b_shift, g, r))

# ============================================================
#  WHITE BACKGROUND
# ============================================================

def clean_background_white(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 51)
    flat = cv2.subtract(gray, bg)
    flat = cv2.normalize(flat, None, 30, 225, cv2.NORM_MINMAX)
    inv = 255 - flat
    inv = cv2.GaussianBlur(inv, (3, 3), 0)
    return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

# ============================================================
#  EMBOSS
# ============================================================

def apply_emboss(img, mode):
    if mode == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mode == 1:
        k = np.array([[0, -1, -1],
                      [1,  0, -1],
                      [1,  1,  0]], dtype=np.float32)
    elif mode == 2:
        k = np.array([[0, -2, -2],
                      [2,  0, -2],
                      [2,  2,  0]], dtype=np.float32)
    else:
        k = np.array([[0, -3, -3],
                      [3,  0, -3],
                      [3,  3,  0]], dtype=np.float32)

    emb = cv2.filter2D(gray, -1, k)
    emb = cv2.normalize(emb, None, 0, 255, cv2.NORM_MINMAX)
    emb = cv2.cvtColor(emb.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(img, 0.7, emb, 0.3, 0)

# ============================================================
#  NEW: INTENSITY CUTOFF (v-mode)
# ============================================================

def apply_intensity_cutoff(img, mode):
    if mode == 0:
        return img

    f = img.astype(np.float32)

    # cutoff percentiles
    if mode == 1:
        low_cut = 0.05
        high_cut = 0.95
        compress = 0.70   # reduce contrast to 70%
    else:
        low_cut = 0.10
        high_cut = 0.90
        compress = 0.50   # reduce contrast to 50%

    lo = np.percentile(f, low_cut * 100)
    hi = np.percentile(f, high_cut * 100)

    # avoid collapse
    if hi - lo < 1:
        return img

    # clip extremes
    f = np.clip(f, lo, hi)

    # compress contrast instead of stretching
    f = (f - lo) / (hi - lo)          # normalize 0–1
    f = f * compress + (1 - compress) / 2   # compress toward mid-gray
    f = f * 255.0

    return f.astype(np.uint8)



# ============================================================
#  SUPER-RESOLUTION
# ============================================================

def capture_superres(cap, dx, dy):
    frames = []

    for _ in range(3):
        ret, f = cap.read()
        if not ret:
            continue

        blue = f[:, :, 0]
        red  = f[:, :, 2]

        red_aligned = apply_subpixel_shift(red, dx, dy)

        phase_float = red_aligned.astype(np.float32) - blue.astype(np.float32)
        phase_norm = cv2.normalize(phase_float, None, 0, 255, cv2.NORM_MINMAX)
        phase = phase_norm.astype(np.uint8)

        if USE_CLAHE:
            phase = clahe.apply(phase)

        out = cv2.cvtColor(phase, cv2.COLOR_GRAY2BGR)
        frames.append(out.astype(np.float32))

    if len(frames) == 0:
        return None

    avg = np.mean(frames, axis=0).astype(np.uint8)
    return avg

# ============================================================
#  RECORDING SYSTEM
# ============================================================

recording = False
record_mode = 0
video_writer = None
last_record_time = 0.0

def start_recording(mode):
    global recording, record_mode, video_writer, last_record_time

    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    filename = f"record_mode{mode}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wmv"

    video_writer = cv2.VideoWriter(filename, fourcc, fps, (1920, 1080))
    recording = True
    record_mode = mode
    last_record_time = 0.0

    print(f"=== Recording started (mode {mode}, playback 25 FPS) ===")

def stop_recording():
    global recording, record_mode, video_writer
    if recording:
        recording = False
        record_mode = 0
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        print("=== Recording stopped ===")

# ============================================================
#  MAIN LOOP
# ============================================================

dx_smooth = 0.0
dy_smooth = 0.0

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    blue = frame[:, :, 0]
    red  = frame[:, :, 2]

    # ROI for phase correlation
    x0, y0, cw, ch = create_center_roi(blue, 0.5)
    blue_roi = blue[y0:y0+ch, x0:x0+cw]
    red_roi  = red[y0:y0+ch, x0:x0+cw]

    small_blue = cv2.resize(blue_roi, PHASECORR_SIZE)
    small_red  = cv2.resize(red_roi,  PHASECORR_SIZE)

    dx_small, dy_small = phase_correlation_shift(small_blue, small_red)

    scale_x = cw / PHASECORR_SIZE[0]
    scale_y = ch / PHASECORR_SIZE[1]

    dx = np.clip(dx_small * scale_x, -MAX_SHIFT_PER_FRAME, MAX_SHIFT_PER_FRAME)
    dy = np.clip(dy_small * scale_y, -MAX_SHIFT_PER_FRAME, MAX_SHIFT_PER_FRAME)

    dx_smooth = (1 - ALPHA) * dx_smooth + ALPHA * dx
    dy_smooth = (1 - ALPHA) * dy_smooth + ALPHA * dy

    # Full-res alignment
    red_aligned = apply_subpixel_shift(red, dx_smooth, dy_smooth)

    # Phase extraction
    phase_float = red_aligned.astype(np.float32) - blue.astype(np.float32)
    phase_norm = cv2.normalize(phase_float, None, 0, 255, cv2.NORM_MINMAX)
    phase = phase_norm.astype(np.uint8)

    if USE_CLAHE:
        phase = clahe.apply(phase)

    if use_noise_reduction:
        phase = cv2.bilateralFilter(phase, 5, 25, 5)

    # NEW: intensity cutoff
    phase = apply_intensity_cutoff(phase, v_mode)

    # ============================================================
    #  IMPROVEMENT MODE (DPR-style sharpening)
    # ============================================================

    if improve_mode > 0 and p_mode in (0, 1):
        blur = cv2.GaussianBlur(phase, (3, 3), 0.6)
        detail = phase.astype(np.float32) - blur.astype(np.float32)

        if improve_mode == 1:
            alpha = 0.15
        elif improve_mode == 2:
            alpha = 0.30
        else:
            alpha = 0.50

        phase = np.clip(phase + alpha * detail, 0, 255).astype(np.uint8)

    # ============================================================
    #  DISPLAY MODES
    # ============================================================

    if p_mode == 0:
        display_img = cv2.cvtColor(phase, cv2.COLOR_GRAY2BGR)

    elif p_mode == 1:
        phase_color = cv2.cvtColor(phase, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frame.astype(np.float32) / 255.0,
                                  0.7,
                                  phase_color.astype(np.float32) / 255.0,
                                  0.3, 0)
        display_img = (blended * 255).astype(np.uint8)

    else:
        display_img = frame.copy()

    # Scale bar
    if show_scale_bar and p_mode != 2:
        UM_PER_PIXEL = 0.083
        SCALE_BAR_UM = 20
        SCALE_BAR_PX = int(SCALE_BAR_UM / UM_PER_PIXEL)
        display_img = draw_scale_bar(display_img, SCALE_BAR_PX, SCALE_BAR_UM)

    # Contrast, background, emboss, averaging
    if p_mode in (0, 1):

        if contrast_mode == 0:
            display_img = process_clean(display_img)
        elif contrast_mode == 1:
            display_img = process_medium(display_img)
        else:
            display_img = process_extreme(display_img)

        if white_background:
            display_img = clean_background_white(display_img)

        display_img = apply_emboss(display_img, emboss_mode)

        avg_buffer.append(display_img.astype(np.float32))
        if len(avg_buffer) > AVG_SIZE:
            avg_buffer.pop(0)
        display_img = np.mean(avg_buffer, axis=0).astype(np.uint8)

    else:
        avg_buffer.clear()

    # On-screen indicators
    cv2.putText(display_img, f"i-mode: {improve_mode}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(display_img, f"v-mode: {v_mode}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Phase Contrast Live", display_img)

    # ============================================================
    #  RECORDING
    # ============================================================

    if recording:
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if record_mode == 1:
            video_writer.write(display_img)

        elif record_mode == 2:
            if current_time - last_record_time >= 1.0 / 3.0:
                video_writer.write(display_img)
                last_record_time = current_time

        elif record_mode == 3:
            if current_time - last_record_time >= 0.1:
                video_writer.write(display_img)
                last_record_time = current_time

    # ============================================================
    #  KEY HANDLING
    # ============================================================

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        stop_recording()
        break

    if key == ord('q'):
        print("Capturing 3-frame super-resolution still...")
        out = capture_superres(cap, dx_smooth, dy_smooth)
        if out is not None:
            filename = f"superres_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            cv2.imwrite(filename, out)
            print("Saved:", filename)
        else:
            print("Super-resolution failed")

    if key == ord('p'):
        p_mode = (p_mode + 1) % 3
        print("P-mode:", p_mode)

    if key == ord('n'):
        use_noise_reduction = not use_noise_reduction
        print("Noise reduction:", use_noise_reduction)

    if key == ord('m'):
        show_scale_bar = not show_scale_bar
        print("Scale bar:", show_scale_bar)

    if key == ord('c'):
        contrast_mode = (contrast_mode + 1) % 3
        print("Contrast mode:", contrast_mode)

    if key == ord('w'):
        white_background = not white_background
        print("White background:", white_background)

    if key == ord('e'):
        emboss_mode = (emboss_mode + 1) % 4
        print("Emboss mode:", emboss_mode)

    if key == ord('i'):
        improve_mode = (improve_mode + 1) % 4
        print("Improve mode:", improve_mode)

    if key == ord('v'):
        v_mode = (v_mode + 1) % 3
        print("V-mode (intensity cutoff):", v_mode)


    if key == ord('s'):
        filename = f"capture_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        cv2.imwrite(filename, display_img)
        print("Saved:", filename)

    # --- Recording mode 1 (25 FPS) ---
    if key == ord('1'):
        if recording and record_mode == 1:
            stop_recording()
        else:
            stop_recording()
            start_recording(1)

    # --- Recording mode 2 (time-lapse 3 FPS capture) ---
    if key == ord('2'):
        if recording and record_mode == 2:
            stop_recording()
        else:
            stop_recording()
            start_recording(2)

    # --- Recording mode 3 (time-lapse 10 FPS capture) ---
    if key == ord('3'):
        if recording and record_mode == 3:
            stop_recording()
        else:
            stop_recording()
            start_recording(3)

# ============================================================
#  CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
