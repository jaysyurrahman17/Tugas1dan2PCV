import numpy as np

# ==========================================
# KONFIGURASI AVATAR
# ==========================================
SCALE_FACTOR = 15

THRESH_EYE_CLOSED = 0.22   
THRESH_MOUTH_OPEN = 0.40   
THRESH_EYE_WIDE = 0.50     

POS_MATA_KANAN = (135, 0) 
POS_MATA_KIRI  = (75, 0)
POS_MULUT      = (75, 25)
POS_MELOTOT    = (135, 45)  

ASSET_FILES = {
    'head':      "gambar/wajah.png",
    'body':      "gambar/badan.png",
    'arm_right': "gambar/tangankanan.png",
    'arm_left':  "gambar/tangankiri.png",
    'leg_right': "gambar/kakikanan.png",
    'leg_left':  "gambar/kakikiri.png",
    'eye_r_close': "gambar/matakanantutup.png",
    'eye_l_close': "gambar/matakiritutup.png",
    'mouth_open':  "gambar/mulutbuka.png",
    'eye_wide':    "gambar/matamelotot.png",
}

BG_PATH = "gambar/background.jpg"
BG_BLUE_TRIGGER = "gambar/background_special.jpg" # Background jika benda biru terdeteksi

# ==========================================
# KONFIGURASI TUGAS 1 (FILTER)
# ==========================================
# Kernel Sharpening (Sesuai Soal)
KERNEL_SHARPEN = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

# ==========================================
# KONFIGURASI TUGAS 2 (WARNA)
# ==========================================
# Rentang warna BIRU dalam HSV
# H: 100-140 biasanya mencakup biru botol/pulpen
BLUE_LOWER = np.array([100, 150, 50])
BLUE_UPPER = np.array([140, 255, 255])