import cv2
import numpy as np
import math
import os

# --- FUNGSI TUGAS 1: FILTERING ---
def create_gaussian_kernel(ksize=5, sigma=1.0):
    """
    Membuat kernel Gaussian 2D secara manual (Syarat Tugas 1).
    Menggunakan Outer Product dari kernel 1D.
    """
    k_1d = cv2.getGaussianKernel(ksize, sigma)
    k_2d = np.outer(k_1d, k_1d)
    return k_2d

def apply_filter_mode(frame, mode):
    """
    Menerapkan filter ke frame berdasarkan mode input keyboard.
    mode '0': Normal
    mode '1': Average Blur
    mode '2': Gaussian Blur (Manual)
    mode '3': Sharpening
    """
    if mode == '1':
        # Average Blur 5x5
        return cv2.blur(frame, (5, 5))
    elif mode == '2':
        # Gaussian Blur Manual 9x9
        kernel = create_gaussian_kernel(9, 1.5)
        return cv2.filter2D(frame, -1, kernel)
    elif mode == '3':
        # Sharpening
        # Import kernel dari config agar rapi, atau definisikan lokal
        k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(frame, -1, k_sharp)
    else:
        return frame

# --- FUNGSI TUGAS 2: DETEKSI WARNA ---
def detect_blue_object(frame, lower_hsv, upper_hsv):
    """
    Mendeteksi keberadaan objek biru dominan.
    Return: (True/False, Masker Gambar)
    """
    # 1. Convert ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Thresholding
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # 3. Morfologi (Pembersihan Noise) - Syarat Tugas 2
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Hapus titik putih kecil
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Tutup lubang hitam
    
    # 4. Cari Kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = False
    if contours:
        # Ambil kontur terbesar
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # Jika area cukup besar (>500 pixel), anggap valid
        if area > 500:
            detected = True
            
    return detected, mask

# --- FUNGSI EXISTING (LOAD GAMBAR DLL) ---
def load_and_scale(path, scale_factor=0.5):
    if not os.path.exists(path):
        if "tutup" in path or "buka" in path or "melotot" in path: return None
        dummy = np.zeros((32, 32, 4), dtype=np.uint8)
        dummy[:] = (255, 0, 255, 255) 
        return dummy
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.shape[2] == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def overlay_image_alpha(bg, fg, x, y):
    if fg is None: return bg
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]
    if x >= w_bg or y >= h_bg: return bg
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w_fg, w_bg), min(y + h_fg, h_bg)
    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1: return bg
    bg_slice = bg[y1:y2, x1:x2]
    fg_slice = fg[fg_y1:fg_y2, fg_x1:fg_x2]
    alpha_fg = fg_slice[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg
    for c in range(0, 3):
        bg_slice[:, :, c] = (alpha_fg * fg_slice[:, :, c] + alpha_bg * bg_slice[:, :, c])
    bg_slice[:, :, 3] = (alpha_fg * fg_slice[:, :, 3] + alpha_bg * bg_slice[:, :, 3])
    bg[y1:y2, x1:x2] = bg_slice
    return bg

def overlay_rotated_image(canvas, img, pivot_point, angle_degrees):
    if img is None: return
    h, w = img.shape[:2]
    center_of_rotation = (w // 2, 0) 
    M = cv2.getRotationMatrix2D(center_of_rotation, -angle_degrees, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center_of_rotation[0]
    M[1, 2] += (new_h / 2) - center_of_rotation[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_NEAREST)
    x_offset = int(pivot_point[0] - (new_w / 2))
    y_offset = int(pivot_point[1] - (new_h / 2))
    y1, y2 = y_offset, y_offset + new_h
    x1, x2 = x_offset, x_offset + new_w
    if y1 < 0: y1 = 0
    if x1 < 0: x1 = 0
    if y2 > canvas.shape[0]: y2 = canvas.shape[0]
    if x2 > canvas.shape[1]: x2 = canvas.shape[1]
    img_y1, img_y2 = y1 - y_offset, y1 - y_offset + (y2 - y1)
    img_x1, img_x2 = x1 - x_offset, x1 - x_offset + (x2 - x1)
    if img_y2 <= img_y1 or img_x2 <= img_x1: return
    slice_canvas = canvas[y1:y2, x1:x2]
    slice_img = rotated[img_y1:img_y2, img_x1:img_x2]
    if slice_img.shape[2] == 4:
        alpha_s = slice_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            slice_canvas[:, :, c] = (alpha_s * slice_img[:, :, c] + alpha_l * slice_canvas[:, :, c])
    else:
        slice_canvas[:] = slice_img 
    canvas[y1:y2, x1:x2] = slice_canvas

def rotate_offset(origin, offset_x, offset_y, angle_degrees, scale):
    rad = math.radians(angle_degrees)
    ox = offset_x * scale
    oy = offset_y * scale
    shift_x = ox * math.cos(rad) - oy * math.sin(rad)
    shift_y = ox * math.sin(rad) + oy * math.cos(rad)
    return (int(origin[0] + shift_x), int(origin[1] + shift_y))

def get_blink_ratio(landmarks, idx_up, idx_down, idx_left, idx_right):
    v_dist = math.dist((landmarks[idx_up].x, landmarks[idx_up].y), (landmarks[idx_down].x, landmarks[idx_down].y))
    h_dist = math.dist((landmarks[idx_left].x, landmarks[idx_left].y), (landmarks[idx_right].x, landmarks[idx_right].y))
    return v_dist / (h_dist + 1e-6)

def get_mouth_ratio(landmarks, idx_top, idx_bottom, idx_left, idx_right):
    v_dist = math.dist((landmarks[idx_top].x, landmarks[idx_top].y), (landmarks[idx_bottom].x, landmarks[idx_bottom].y))
    h_dist = math.dist((landmarks[idx_left].x, landmarks[idx_left].y), (landmarks[idx_right].x, landmarks[idx_right].y))
    return v_dist / (h_dist + 1e-6)

def calc_angle(p1, p2):
    dy, dx = p2[1] - p1[1], p2[0] - p1[0]
    return math.degrees(math.atan2(dy, dx)) - 90