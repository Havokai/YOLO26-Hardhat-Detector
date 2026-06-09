import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


class Track:
    def __init__(self, track_id, bbox, class_id, max_age=60, n_init=2):
        self.track_id = track_id
        self.class_id = class_id
        self.last_class_id = class_id
        self.max_age = max_age
        self.n_init = n_init
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False
        self.last_known_bbox = [int(v) for v in bbox]  # ← ДОБАВЛЕНО

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(4, 1)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, class_id=None):
        self.kf.update(np.array(bbox).reshape(4, 1))  # ← ИСПРАВЛЕНО
        self.hits += 1
        self.time_since_update = 0
        if class_id is not None:
            self.class_id = class_id
            self.last_class_id = class_id
        if self.hits >= self.n_init:
            self.confirmed = True
        self.last_known_bbox = [int(v) for v in bbox]  # ← ДОБАВЛЕНО

    def get_state(self):
        state = self.kf.x[:4].flatten()
        kalman_bbox = [int(state[0]), int(state[1]), int(state[2]), int(state[3])]

        # Если трек не обновлялся — смешиваем предсказание Калмана с последним известным положением
        if self.time_since_update > 3:
            # Коэффициент затухания: чем дольше потерян, тем ближе к last_known
            alpha = min(1.0, (self.time_since_update - 3) / 10.0)  # Плавно от 0 до 1 за 10 кадров

            # Интерполяция между предсказанием Калмана (движение) и последним известным (якорь)
            result = [
                int((1 - alpha) * kalman_bbox[0] + alpha * self.last_known_bbox[0]),
                int((1 - alpha) * kalman_bbox[1] + alpha * self.last_known_bbox[1]),
                int((1 - alpha) * kalman_bbox[2] + alpha * self.last_known_bbox[2]),
                int((1 - alpha) * kalman_bbox[3] + alpha * self.last_known_bbox[3]),
            ]
            return result

        return kalman_bbox



    def is_deleted(self):
        return self.time_since_update > self.max_age


class Tracker:

    def __init__(self, max_age=60, n_init=2, max_iou_distance=0.5):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.tracks = []
        self.next_id = 0

    def update(self, detections, frame=None):
        # Предсказание новых позиций
        for track in self.tracks:
            track.predict()

        # Сопоставление треков и обнаружений
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # Обновление сопоставленных треков
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            track.update(det['bbox'], class_id=det['class_id'])

        # Создание новых треков для несопоставленных обнаружений
        for det_idx in unmatched_dets:
            det = detections[det_idx]

            # Проверяем, нет ли уже трека в этом месте
            overlapping_track = self._is_overlapping(det['bbox'], iou_threshold=0.35)

            if overlapping_track is not None:
                # Обновляем существующий трек вместо создания нового
                overlapping_track.update(det['bbox'], class_id=det['class_id'])
            else:
                # Создаём новый трек
                new_track = Track(self.next_id, det['bbox'], det['class_id'],
                                  self.max_age, self.n_init)
                self.tracks.append(new_track)
                self.next_id += 1

        # Удаление устаревших треков
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Формирование результата
        active_tracks = []
        for track in self.tracks:
            if track.confirmed:
                bbox = track.get_state()
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': bbox,
                    'class_id': track.class_id
                })
        return active_tracks

    def _match(self, detections):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Матрица IoU
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t_idx, track in enumerate(self.tracks):
            t_bbox = track.get_state()
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = iou(t_bbox, det['bbox'])

        # Венгерский алгоритм
        cost_matrix = 1 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= 1 - self.max_iou_distance:
                matched.append((r, c))
                unmatched_dets.remove(c)
                unmatched_tracks.remove(r)

        return matched, unmatched_dets, unmatched_tracks

    def _is_overlapping(self, new_bbox, iou_threshold=0.65):

        for track in self.tracks:
            if not track.confirmed:
                continue
            existing_bbox = track.get_state()
            overlap = iou(new_bbox, existing_bbox)
            if overlap > iou_threshold:
                return track  # Возвращаем трек, с которым перекрывается
        return None

