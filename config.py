
import os

# --- Пути ---
MODEL_PATH = 'best_yolo26n_aug.pt'
VIDEO_SOURCE = 0  # 0 — веб-камера, путь к файлу или URL

# --- Устройство ---
DEVICE = 'cuda'  # 'cuda' или 'cpu'
CONF_THRESHOLD = 0.5

VIDEO_FILE_PATH = 't4.mp4'

# --- Трекер ---
TRACKER_TYPE = "kalman"

# --- Параметры трекеров ---
TRACKER_PARAMS = {
    "kalman": {
        "max_age": 20,
        "n_init": 2,
        "max_iou_distance": 0.35,
    }
}
DETECTION_MODE = "cascade"  # "cascade" или "direct"
# --- Тревога ---
VIOLATION_THRESHOLD_SEC = 2
ALERT_COOLDOWN_SEC = 10.0