import time
import os
import cv2
from collections import defaultdict


class AlertManager:
    CLASS_HELMET = 0
    CLASS_NO_HELMET = 1

    def __init__(self, violation_threshold_sec=2.0, cooldown_sec=5.0, save_dir="violations_screenshots"):
        self.violation_threshold = violation_threshold_sec
        self.cooldown = cooldown_sec
        self.save_dir = save_dir
        self.last_frame = None
        self.source_label = "cam0"  # метка источника для скриншотов
        os.makedirs(self.save_dir, exist_ok=True)

        self.track_state = defaultdict(lambda: {
            'no_helmet_start': None,
            'last_alert': 0.0,
            'was_violator': False,
            'is_active': False,
            'last_screenshot': 0.0,
        })

    def set_frame(self, frame):
        self.last_frame = frame

    def set_source_label(self, label):
        # Установить метку источника (камера/файл)
        self.source_label = label

    def update(self, tracked_objects, current_time):
        violations = []
        active_ids = {obj['track_id'] for obj in tracked_objects}

        for tid in list(self.track_state.keys()):
            if tid not in active_ids:
                self.track_state[tid]['no_helmet_start'] = None
                self.track_state[tid]['is_active'] = False

        for obj in tracked_objects:
            tid = obj['track_id']
            cls = obj['class_id']
            state = self.track_state[tid]
            state['is_active'] = True

            if cls == self.CLASS_NO_HELMET:
                if state['no_helmet_start'] is None:
                    state['no_helmet_start'] = current_time
                else:
                    duration = current_time - state['no_helmet_start']
                    if duration >= self.violation_threshold:
                        if current_time - state['last_alert'] >= self.cooldown:
                            state['last_alert'] = current_time
                            state['was_violator'] = True
                            print(f"НАРУШЕНИЕ! Трек #{tid}: без каски ({duration:.1f}с)")
                            self._save_screenshot(tid, obj['bbox'], duration, current_time)
                        violations.append(tid)
            else:
                state['no_helmet_start'] = None

        return violations

    def _save_screenshot(self, track_id, bbox, duration, current_time):
        if self.last_frame is None:
            return

        state = self.track_state[track_id]
        if current_time - state['last_screenshot'] < 5.0:
            return
        state['last_screenshot'] = current_time

        frame = self.last_frame.copy()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Красная рамка
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Подпись: источник + ID + класс + длительность
        label = f"[{self.source_label}] ID:{track_id} NO HELMET ({duration:.0f}s)"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - lh - baseline - 8), (x1 + lw + 8, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 4, y1 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # Дата и время
        datetime_str = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime(current_time))
        (tw, th), _ = cv2.getTextSize(datetime_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, h - th - 20), (tw + 25, h - 5), (0, 0, 0), -1)
        cv2.putText(frame, datetime_str, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # Сохраняем
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.save_dir,
            f"{self.source_label}_id{track_id}_{timestamp}_d{duration:.0f}s.jpg"
        )
        cv2.imwrite(filename, frame)
        print(f"Скриншот сохранён: {filename}")

    def get_active_violators(self, tracked_objects, current_time):
        active = []
        for obj in tracked_objects:
            tid = obj['track_id']
            state = self.track_state[tid]
            if (obj['class_id'] == self.CLASS_NO_HELMET and
                    state['no_helmet_start'] is not None):
                duration = current_time - state['no_helmet_start']
                if duration >= self.violation_threshold:
                    active.append({
                        'track_id': tid,
                        'duration': duration,
                        'bbox': obj['bbox']
                    })
        return active

    def get_statistics(self):
        total = len(self.track_state)
        violators = sum(1 for s in self.track_state.values() if s['was_violator'])
        return {
            'total_tracks': total,
            'violators': violators,
            'compliance_rate': 100 * (1 - violators / total) if total > 0 else 100
        }