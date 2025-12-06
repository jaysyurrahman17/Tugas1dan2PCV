import cv2
import mediapipe as mp
import numpy as np
import math
import os

# Import modul buatan sendiri
import config as cfg
import utils

# ==========================================
# SETUP
# ==========================================
print("Memuat aset...")

assets = {}
for key, filename in cfg.ASSET_FILES.items():
    assets[key] = utils.load_and_scale(filename, cfg.SCALE_FACTOR)

# Load Background Normal
bg_normal = None
if os.path.exists(cfg.BG_PATH):
    bg_normal = cv2.imread(cfg.BG_PATH)
else:
    print(f"⚠️ {cfg.BG_PATH} tidak ada, pakai putih.")

# Load Background Spesial (Untuk Aksi Tugas 2)
bg_special = None
if os.path.exists(cfg.BG_BLUE_TRIGGER):
    print("✅ Background Spesial ditemukan!")
    bg_special = cv2.imread(cfg.BG_BLUE_TRIGGER)
else:
    # Kalau tidak ada file gambar, kita buat warna solid Merah
    print("⚠️ Background spesial tidak ada, nanti pakai warna merah solid.")

# STATE VARIABLES
filter_mode = '0' # '0'=Normal, '1'=Avg, '2'=Gauss, '3'=Sharp
blue_detected = False

# ==========================================
# MAIN LOOP
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n=== KONTROL TUGAS 1 ===")
print("Tekan 1: Average Blur")
print("Tekan 2: Gaussian Blur")
print("Tekan 3: Sharpening")
print("Tekan 0: Reset Normal")
print("\n=== INSTRUKSI TUGAS 2 ===")
print("Tunjukkan benda berwarna BIRU untuk mengubah Background!")

with mp_holistic.Holistic(model_complexity=1, refine_face_landmarks=True) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # -----------------------------------------------------------
        # TUGAS 2: DETEKSI OBJEK BIRU (PEMICU AKSI)
        # -----------------------------------------------------------
        is_blue, mask_blue = utils.detect_blue_object(frame, cfg.BLUE_LOWER, cfg.BLUE_UPPER)
        
        # LOGIKA GANTI BACKGROUND (AKSI TUGAS 2)
        # Jika biru terdeteksi, pakai bg_special. Jika tidak, pakai bg_normal.
        current_bg = None
        
        if is_blue:
            if bg_special is not None:
                current_bg = cv2.resize(bg_special, (w, h))
            else:
                # Fallback: Layar Merah Solid jika tidak ada gambar
                current_bg = np.zeros((h, w, 3), dtype=np.uint8)
                current_bg[:] = (0, 0, 255) # Merah (BGR)
            
            # Tambahkan teks notifikasi
            cv2.putText(frame, "OBJEK BIRU TERDETEKSI!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            if bg_normal is not None:
                current_bg = cv2.resize(bg_normal, (w, h))
            else:
                # Fallback: Layar Putih
                current_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Mulai gambar avatar di atas current_bg
        avatar = current_bg.copy()

        # -----------------------------------------------------------
        # MEDIAPIPE & LOGIKA AVATAR (CORE PROJECT)
        # -----------------------------------------------------------
        # NOTE: MediaPipe butuh gambar bersih. JANGAN diblur dulu sebelum masuk sini.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        if results.pose_landmarks:
             mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            GESER_GLOBAL_Y = 0 
            def get_pt(idx):
                lm = results.pose_landmarks.landmark[idx]
                return (int(lm.x * w), int(lm.y * h) + GESER_GLOBAL_Y)
            
            p_nose = get_pt(0)
            p_sh_l, p_sh_r = get_pt(11), get_pt(12)
            p_elbow_l = get_pt(13)
            p_wrist_l, p_wrist_r = get_pt(15), get_pt(16)
            p_ankle_l, p_ankle_r = get_pt(27), get_pt(28)
            p_hip_l, p_hip_r = get_pt(23), get_pt(24)

            center_sh = ((p_sh_l[0] + p_sh_r[0]) // 2, (p_sh_l[1] + p_sh_r[1]) // 2)
            center_hips = ((p_hip_l[0] + p_hip_r[0]) // 2, (p_hip_l[1] + p_hip_r[1]) // 2)
            shoulder_width = math.dist(p_sh_l, p_sh_r)
            
            base_scale = shoulder_width / 150.0 
            body_tilt = utils.calc_angle(center_sh, center_hips) 

            pos_head  = utils.rotate_offset(center_sh, 0, -119, body_tilt, base_scale)
            pos_leg_r = utils.rotate_offset(center_sh, 32, 170, body_tilt, base_scale)
            pos_leg_l = utils.rotate_offset(center_sh, -32, 170, body_tilt, base_scale)
            pos_arm_r = utils.rotate_offset(center_sh, -85, 0, body_tilt, base_scale)
            pos_arm_l = utils.rotate_offset(center_sh, 85, 0, body_tilt, base_scale)

            def render_part(img_source, pivot, angle):
                if img_source is None: return
                cur_h, cur_w = img_source.shape[:2]
                dyn_w = max(1, int(cur_w * base_scale))
                dyn_h = max(1, int(cur_h * base_scale))
                img_zoomed = cv2.resize(img_source, (dyn_w, dyn_h), interpolation=cv2.INTER_NEAREST)
                utils.overlay_rotated_image(avatar, img_zoomed, pivot, angle)

            render_part(assets['leg_right'], pos_leg_r, utils.calc_angle(p_hip_l, p_ankle_l))
            render_part(assets['leg_left'], pos_leg_l, utils.calc_angle(p_hip_r, p_ankle_r))
            render_part(assets['body'], center_sh, body_tilt)

            current_head_img = assets['head'].copy() 
            if results.face_landmarks:
                fl = results.face_landmarks.landmark
                l_eye = utils.get_blink_ratio(fl, 159, 145, 33, 133)
                r_eye = utils.get_blink_ratio(fl, 386, 374, 362, 263)
                mouth = utils.get_mouth_ratio(fl, 13, 14, 78, 308)
                
                is_melotot = (r_eye > cfg.THRESH_EYE_WIDE) and (l_eye > cfg.THRESH_EYE_WIDE)

                if is_melotot:
                    current_head_img = utils.overlay_image_alpha(
                        current_head_img, assets['eye_wide'], cfg.POS_MELOTOT[0], cfg.POS_MELOTOT[1])
                else:
                    if r_eye < cfg.THRESH_EYE_CLOSED:
                        current_head_img = utils.overlay_image_alpha(
                            current_head_img, assets['eye_r_close'], cfg.POS_MATA_KANAN[0], cfg.POS_MATA_KANAN[1])
                    if l_eye < cfg.THRESH_EYE_CLOSED:
                        current_head_img = utils.overlay_image_alpha(
                            current_head_img, assets['eye_l_close'], cfg.POS_MATA_KIRI[0], cfg.POS_MATA_KIRI[1])

                if mouth > cfg.THRESH_MOUTH_OPEN:
                    current_head_img = utils.overlay_image_alpha(
                        current_head_img, assets['mouth_open'], cfg.POS_MULUT[0], cfg.POS_MULUT[1])
            
            render_part(current_head_img, pos_head, body_tilt)
            render_part(assets['arm_right'], pos_arm_r, utils.calc_angle(p_sh_r, p_wrist_r))
            render_part(assets['arm_left'], pos_arm_l, utils.calc_angle(p_sh_l, p_wrist_l))

            hand_above_head = p_wrist_l[1] < p_nose[1]
            elbow_above_shoulder = p_elbow_l[1] < p_sh_l[1]
            if hand_above_head and elbow_above_shoulder:
                cv2.putText(avatar, "NEIN!!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                            3.0, (0, 0, 255), 5, cv2.LINE_AA)

        # -----------------------------------------------------------
        # TUGAS 1: APPLY FILTER KE OUTPUT KAMERA
        # -----------------------------------------------------------
        # Kita terapkan filter ke 'frame' (kamera asli) untuk ditampilkan di jendela kanan
        # Ini memenuhi syarat: "Terapkan filter pada video input"
        
        display_frame = utils.apply_filter_mode(frame, filter_mode)
        
        # Tambahkan teks status mode
        mode_names = {'0': "Normal", '1': "Avg Blur", '2': "Gauss Blur", '3': "Sharpen"}
        cv2.putText(display_frame, f"Filter: {mode_names.get(filter_mode, 'Normal')}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Tampilkan Jendela
        cv2.imshow('VTuber Output (Tugas 2 Trigger)', avatar)
        cv2.imshow('Kamera Input (Tugas 1 Filter)', display_frame) 
        
        # KEYBOARD CONTROL
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'): break
        elif key == ord('0'): filter_mode = '0'
        elif key == ord('1'): filter_mode = '1'
        elif key == ord('2'): filter_mode = '2'
        elif key == ord('3'): filter_mode = '3'

cap.release()
cv2.destroyAllWindows()