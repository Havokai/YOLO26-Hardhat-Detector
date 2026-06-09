import cv2
import time
import argparse
import torch
import numpy as np
from detector import Detector
from alert_manager import AlertManager
import config
from tracker_module import Tracker as KalmanTracker
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "0"


def open_camera(source, width=1920, height=1080):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Камера {source}")
    return cap


class DummyTracker:
    def __init__(self):
        self.next_id = 0

    def update(self, detections, frame=None):
        results = []
        for det in detections:
            results.append({
                'track_id': self.next_id,
                'bbox': det['bbox'],
                'class_id': det['class_id']
            })
            self.next_id += 1
        return results


def iou_simple(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def create_tracker():
    tp = config.TRACKER_TYPE
    params = config.TRACKER_PARAMS.get(tp, {})
    if tp == "kalman":
        return KalmanTracker(
            max_age=params.get("max_age", 10),
            n_init=params.get("n_init", 1),
            max_iou_distance=params.get("max_iou_distance", 0.3),
        )
    else:
        return DummyTracker()


def process_frame(frame, detector, tracker, alert_manager, current_time):
    alert_manager.set_frame(frame)
    detections = detector.detect(frame)
    tracked_objects = tracker.update(detections, frame)
    violations = alert_manager.update(tracked_objects, current_time)
    return tracked_objects, violations, detections


def draw_results(frame, tracked_objects, violations, active_violators=None, detections=None):
    class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}
    class_labels = {0: 'HELMET', 1: 'NO HELMET'}

    violation_ids = set()
    if violations:
        for v in violations:
            if isinstance(v, dict):
                violation_ids.add(v.get('track_id'))
            else:
                violation_ids.add(v)

    active_ids = set()
    active_durations = {}
    if active_violators:
        for v in active_violators:
            tid = v.get('track_id')
            active_ids.add(tid)
            if 'duration' in v:
                active_durations[tid] = v['duration']

    for obj in tracked_objects:
        l, t, r, b = obj['bbox']
        tid = obj['track_id']
        cls = obj['class_id']
        if cls is None or cls == -1:
            cls = 1

        conf = obj.get('confidence', None)
        if conf is None and detections:
            for det in detections:
                if iou_simple(obj['bbox'], det['bbox']) > 0.5:
                    conf = det['confidence']
                    break

        is_violator = tid in violation_ids
        is_active = tid in active_ids

        if is_violator:
            color = (0, 80, 255)
            color_bg = (0, 40, 180)
        elif cls == 0:
            color = (0, 220, 80)
            color_bg = (0, 140, 50)
        else:
            color = (60, 60, 255)
            color_bg = (40, 40, 180)

        thickness = 2
        corner = 8

        cv2.line(frame, (l + corner, t), (r - corner, t), color, thickness)
        cv2.line(frame, (r, t + corner), (r, b - corner), color, thickness)
        cv2.line(frame, (l + corner, b), (r - corner, b), color, thickness)
        cv2.line(frame, (l, t + corner), (l, b - corner), color, thickness)

        cv2.ellipse(frame, (l + corner, t + corner), (corner, corner), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (r - corner, t + corner), (corner, corner), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (l + corner, b - corner), (corner, corner), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (r - corner, b - corner), (corner, corner), 0, 0, 90, color, thickness)

        if is_active and tid in active_durations:
            label = f"ID:{tid}  NO HELMET  {active_durations[tid]:.0f}s"
        elif is_violator:
            label = f"ID:{tid}  NO HELMET"
        elif conf is not None:
            label = f"ID:{tid}  {class_labels.get(cls, '?')}  {conf:.0%}"
        else:
            label = f"ID:{tid}  {class_labels.get(cls, '?')}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, 2)
        pad = 6

        px1, py1 = l, t - lh - baseline - pad * 2
        px2, py2 = l + lw + pad * 2, t

        cv2.rectangle(frame, (px1 + corner, py1), (px2 - corner, py2), color_bg, -1)
        cv2.rectangle(frame, (px1, py1 + corner), (px2, py2 - corner), color_bg, -1)
        cv2.circle(frame, (px1 + corner, py1 + corner), corner, color_bg, -1)
        cv2.circle(frame, (px2 - corner, py1 + corner), corner, color_bg, -1)
        cv2.circle(frame, (px1 + corner, py2 - corner), corner, color_bg, -1)
        cv2.circle(frame, (px2 - corner, py2 - corner), corner, color_bg, -1)

        cv2.putText(frame, label, (l + pad, t - baseline - pad), font, font_scale, (255, 255, 255), 2)

    helmet_ok = sum(1 for o in tracked_objects if o['class_id'] == 0)
    no_helmet = sum(1 for o in tracked_objects if o['class_id'] == 1)
    active_count = len(active_ids)

    h, w = frame.shape[:2]
    bar_h = 35
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    y = h - bar_h + 22
    cv2.putText(frame,
                f"With: {helmet_ok}    Without: {no_helmet}    Violators: {active_count}    Alerts: {len(violation_ids)}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    return frame


def nothing(x):
    pass


def auto_detect_cameras():
    print("\nПоиск камер...")
    cameras = []
    i = 0
    consecutive_fails = 0

    while consecutive_fails < 3 and i < 20:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append(i)
                print(f"  ✓ Камера {i}: {w}x{h}")
                consecutive_fails = 0
            else:
                consecutive_fails += 1
        else:
            consecutive_fails += 1
        cap.release()
        i += 1

    if not cameras:
        print("  Камеры не найдены, использую индекс 0")
        cameras = [0]

    print(f"Найдено камер: {len(cameras)} — {cameras}\n")
    return cameras


def create_control_window(num_cameras=1):
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Control Panel', 480, 760)

    cv2.createTrackbar('Confidence', 'Control Panel', 35, 100, nothing)
    cv2.createTrackbar('Source', 'Control Panel', 0, 1, nothing)
    cv2.createTrackbar('Model', 'Control Panel', 0, 2, nothing)
    cv2.createTrackbar('Tracker', 'Control Panel', 0, 1, nothing)
    cv2.createTrackbar('Detection', 'Control Panel', 0, 1, nothing)

    if num_cameras > 1:
        cv2.createTrackbar('Camera', 'Control Panel', 0, num_cameras - 1, nothing)

    panel = np.zeros((760, 480, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    return panel


import cv2


def draw_card(img, x1, y1, x2, y2,
              color=(55, 55, 65),
              border=(80, 80, 95)):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)


def draw_progress(img, x, y, w, h, value,
                  color=(0, 255, 136)):

    cv2.rectangle(img, (x, y), (x + w, y + h),
                  (70, 70, 70), -1)

    fill = int(w * max(0, min(1, value)))

    cv2.rectangle(img,
                  (x, y),
                  (x + fill, y + h),
                  color,
                  -1)

    cv2.rectangle(img,
                  (x, y),
                  (x + w, y + h),
                  (120, 120, 120),
                  1)


def draw_control_panel(panel, state):

    # Dark theme
    panel[:] = (28, 28, 32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_big = cv2.FONT_HERSHEY_DUPLEX

    width = panel.shape[1]

    # HEADER
    cv2.putText(panel,
                "HELMET DETECTOR",
                (15, 35),
                font_big,
                0.8,
                (0, 255, 136),
                2)

    # FPS CARD
    draw_card(panel, 10, 80, width - 10, 170)

    fps = state.get("fps", 0)

    cv2.putText(panel,
                f"{fps:.1f}",
                (20, 145),
                font_big,
                2.0,
                (0, 255, 255),
                3)

    cv2.putText(panel,
                "FPS",
                (170, 145),
                font_big,
                0.9,
                (180, 180, 180),
                2)


    # STATUS CARD
    draw_card(panel, 10, 185, width - 10, 245)

    running = not state.get("paused", False)

    status_text = "RUNNING" if running else "PAUSED"

    status_color = (
        (0, 255, 0)
        if running
        else (0, 165, 255)
    )

    cv2.circle(panel,
               (30, 215),
               10,
               status_color,
               -1)

    cv2.putText(panel,
                status_text,
                (55, 223),
                font_big,
                0.8,
                status_color,
                2)


    # CONFIDENCE
    draw_card(panel, 10, 260, width - 10, 335)

    conf = state.get("conf", 0.5)

    cv2.putText(panel,
                f"Confidence {conf:.2f}",
                (20, 290),
                font,
                0.6,
                (220, 220, 220),
                2)

    draw_progress(panel,
                  20,
                  305,
                  width - 60,
                  15,
                  conf)


    # SYSTEM INFO
    draw_card(panel, 10, 350, width - 10, 520)

    y = 380

    source = "Webcam" if state.get("source_type", 0) == 0 else "Video"

    cv2.putText(panel,
                f"Source: {source}",
                (20, y),
                font,
                0.55,
                (220, 220, 220),
                2)

    y += 30

    if state.get("source_type", 0) == 0:
        cv2.putText(panel,
                    f"Camera: {state.get('camera_idx', 0)}",
                    (20, y),
                    font,
                    0.55,
                    (220, 220, 220),
                    2)

        y += 30

    model = state.get("model_name", "YOLO26")

    if len(model) > 28:
        model = "..." + model[-25:]

    cv2.putText(panel,
                f"Model: {model}",
                (20, y),
                font,
                0.5,
                (220, 220, 220),
                2)

    y += 30

    tracker = "Kalman" if state.get("tracker_type", 0) == 0 else "Dummy"

    cv2.putText(panel,
                f"Tracker: {tracker}",
                (20, y),
                font,
                0.55,
                (220, 220, 220),
                2)

    y += 30

    mode = "Cascade" if state.get("detection_mode", 0) == 0 else "Direct"

    cv2.putText(panel,
                f"Detection: {mode}",
                (20, y),
                font,
                0.55,
                (220, 220, 220),
                2)


    # DETECTIONS
    draw_card(panel, 10, 535, width - 10, 675)

    helmet = state.get("helmet", 0)
    no_helmet = state.get("no_helmet", 0)
    violators = state.get("violations", 0)


    cv2.circle(panel, (28, 565), 8, (0, 255, 0), -1)
    cv2.putText(panel,
                f"Helmet: {helmet}",
                (45, 570),
                font,
                0.65,
                (255, 255, 255),
                2)

    cv2.circle(panel, (28, 605), 8, (0, 0, 255), -1)
    cv2.putText(panel,
                f"No helmet: {no_helmet}",
                (45, 610),
                font,
                0.65,
                (255, 255, 255),
                2)

    cv2.circle(panel, (28, 645), 8, (0, 165, 255), -1)
    cv2.putText(panel,
                f"Violators: {violators}",
                (45, 650),
                font,
                0.65,
                (255, 255, 255),
                2)



    # FOOTER
    cv2.putText(panel,
                "Q Exit | SPACE Pause | T Tracker",
                (10, panel.shape[0] - 35),
                font,
                0.45,
                (150, 150, 150),
                1)

    cv2.putText(panel,
                "D Detection | C Camera",
                (10, panel.shape[0] - 15),
                font,
                0.45,
                (150, 150, 150),
                1)

    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=str(config.VIDEO_SOURCE))
    parser.add_argument('--model', type=str, default=config.MODEL_PATH)
    parser.add_argument('--device', type=str, default=config.DEVICE)
    parser.add_argument('--conf', type=float, default=config.CONF_THRESHOLD)
    args = parser.parse_args()

    available_cameras = auto_detect_cameras()

    source = args.source
    if str(source).isdigit():
        source = int(source)

    if isinstance(source, dict):
        source = source['index']

    if isinstance(source, int) and source not in available_cameras:
        source = available_cameras[0]

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    models = ["best_yolo26n_aug.pt", "best_yolo26s_aug.pt", "best_yolo26m_aug.pt"]
    model_idx = 0
    model_path = args.model if args.model else models[model_idx]

    print("=" * 50)
    print("СИСТЕМА ДЕТЕКЦИИ ЗАЩИТНЫХ КАСОК")
    print("=" * 50)
    print(f"Модель: {model_path}")

    cap = open_camera(source)
    video_fps = cap.get(cv2.CAP_PROP_FPS)


    detector = Detector(
        model_path=model_path,
        conf_threshold=args.conf,
        device=args.device,
        cascade=(config.DETECTION_MODE == "cascade")
    )
    tracker = create_tracker()
    alert_manager = AlertManager(
        violation_threshold_sec=config.VIOLATION_THRESHOLD_SEC,
        cooldown_sec=config.ALERT_COOLDOWN_SEC
    )
    alert_manager.set_source_label("Webcam" if source == 0 else os.path.basename(str(source)))

    cv2.namedWindow('Helmet Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Helmet Detection', 960, 540)

    num_cameras = len(available_cameras)
    control_panel = create_control_window(num_cameras)

    frame_count = 0
    fps_start = time.time()
    paused = False
    source_type = 0
    tracker_type = 0
    detection_mode = 0 if config.DETECTION_MODE == "cascade" else 1
    camera_idx = 0
    current_source = source
    current_video_file = config.VIDEO_FILE_PATH

    state = {
        'conf': args.conf,
        'fps': 0,
        'paused': False,
        'helmet': 0,
        'no_helmet': 0,
        'violations': 0,
        'source_type': source_type,
        'model_name': os.path.basename(model_path),
        'tracker_type': tracker_type,
        'detection_mode': detection_mode,
        'camera_idx': camera_idx,
        'num_cameras': num_cameras,
        'video_file': os.path.basename(current_video_file) if source_type == 1 else 'N/A',
    }

    cv2.setTrackbarPos('Confidence', 'Control Panel', int(args.conf * 100))
    cv2.setTrackbarPos('Source', 'Control Panel', source_type)
    cv2.setTrackbarPos('Model', 'Control Panel', model_idx)
    cv2.setTrackbarPos('Tracker', 'Control Panel', tracker_type)
    cv2.setTrackbarPos('Detection', 'Control Panel', detection_mode)
    if num_cameras > 1:
        cv2.setTrackbarPos('Camera', 'Control Panel', camera_idx)

    panel_img = np.zeros((760, 480, 3), dtype=np.uint8)

    print("\nУправление: Q-выход, ПРОБЕЛ-пауза, T-смена трекера, D-смена детекции, C-следующая камера")
    print(f"Видеофайл (по умолчанию): {config.VIDEO_FILE_PATH}")

    try:
        while True:
            conf_val = cv2.getTrackbarPos('Confidence', 'Control Panel') / 100.0
            source_val = cv2.getTrackbarPos('Source', 'Control Panel')
            model_val = cv2.getTrackbarPos('Model', 'Control Panel')
            tracker_val = cv2.getTrackbarPos('Tracker', 'Control Panel')
            detection_val = cv2.getTrackbarPos('Detection', 'Control Panel')
            cam_val = cv2.getTrackbarPos('Camera', 'Control Panel') if num_cameras > 1 else 0

            if conf_val != args.conf:
                args.conf = conf_val
                detector.conf_threshold = conf_val
                state['conf'] = conf_val

            if model_val != model_idx:
                model_idx = model_val
                model_path = models[model_idx]
                detector = Detector(model_path=model_path, conf_threshold=args.conf, device=args.device)
                state['model_name'] = os.path.basename(model_path)
                print(f"Модель: {model_path}")

            if tracker_val != tracker_type:
                tracker_type = tracker_val
                if tracker_type == 0:
                    config.TRACKER_TYPE = "kalman"
                elif tracker_type == 1:
                    config.TRACKER_TYPE = "dummy"
                tracker = create_tracker()
                state['tracker_type'] = tracker_type

            if detection_val != detection_mode:
                detection_mode = detection_val
                state['detection_mode'] = detection_mode
                if detection_mode == 0:
                    detector.set_cascade(True)
                    config.DETECTION_MODE = "cascade"
                else:
                    detector.set_cascade(False)
                    config.DETECTION_MODE = "direct"
                print(f"Метод детекции: {'Каскадный' if detection_mode == 0 else 'Прямой'}")

            if cam_val != camera_idx and source_type == 0:
                camera_idx = cam_val
                cap.release()
                cap = open_camera(available_cameras[camera_idx])
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                alert_manager.set_source_label(f"Cam {available_cameras[camera_idx]}")
                state['camera_idx'] = camera_idx
                frame_count = 0
                state['fps'] = 0

            if source_val != source_type:
                source_type = source_val
                cap.release()
                if source_type == 0:
                    current_source = available_cameras[camera_idx]
                    state['video_file'] = 'N/A'
                    cap = open_camera(current_source)
                    alert_manager.set_source_label(f"Cam {current_source}")
                else:
                    current_source = config.VIDEO_FILE_PATH
                    state['video_file'] = os.path.basename(current_source)
                    cap = cv2.VideoCapture(current_source)
                    alert_manager.set_source_label(os.path.basename(current_source))
                state['source_type'] = source_type
                frame_count = 0
                state['fps'] = 0

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                current_time = time.time()

                tracked_objects, violations, detections = process_frame(
                    frame, detector, tracker, alert_manager, current_time
                )

                active_violators = alert_manager.get_active_violators(tracked_objects, current_time)
                frame = draw_results(frame, tracked_objects, violations, active_violators, detections)

                state['helmet'] = sum(1 for o in tracked_objects if o['class_id'] == 0)
                state['no_helmet'] = sum(1 for o in tracked_objects if o['class_id'] == 1)
                state['violations'] = len(violations) if violations else 0

                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    state['fps'] = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                cv2.imshow('Helmet Detection', frame)

            state['paused'] = paused
            panel_img = draw_control_panel(panel_img, state)
            cv2.imshow('Control Panel', panel_img)

            wait_ms = 1
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d'):
                detection_mode = 1 - detection_mode
                cv2.setTrackbarPos('Detection', 'Control Panel', detection_mode)
                if detection_mode == 0:
                    detector.set_cascade(True)
                    config.DETECTION_MODE = "cascade"
                else:
                    detector.set_cascade(False)
                    config.DETECTION_MODE = "direct"
                state['detection_mode'] = detection_mode
                print(f"Метод детекции: {'Каскадный' if detection_mode == 0 else 'Прямой'}")
            elif key == ord('t'):
                if config.TRACKER_TYPE == "kalman":
                    config.TRACKER_TYPE = "dummy"
                    tracker_type = 1
                else:
                    config.TRACKER_TYPE = "kalman"
                    tracker_type = 0
                tracker = create_tracker()
                cv2.setTrackbarPos('Tracker', 'Control Panel', tracker_type)
                state['tracker_type'] = tracker_type
                print(f"Трекер: {config.TRACKER_TYPE}")
            elif key == ord('c') and source_type == 0 and num_cameras > 1:
                camera_idx = (camera_idx + 1) % num_cameras
                cv2.setTrackbarPos('Camera', 'Control Panel', camera_idx)
                cap.release()
                cap = open_camera(available_cameras[camera_idx])


                alert_manager.set_source_label(f"Cam {available_cameras[camera_idx]}")
                state['camera_idx'] = camera_idx
                frame_count = 0
                state['fps'] = 0

    except KeyboardInterrupt:
        print("\nОстановка.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stats = alert_manager.get_statistics()
        print("\n" + "=" * 50)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 50)
        print(f"Всего треков:    {stats['total_tracks']}")
        print(f"Нарушителей:     {stats['violators']}")
        print("=" * 50)


if __name__ == '__main__':
    main()