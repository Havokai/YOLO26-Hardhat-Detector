from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='best_yolo26n_aug.pt', conf_threshold=0.5, device='cuda',
                 cascade=True):
        self.model = YOLO(model_path)           # Модель касок
        self.person_model = YOLO('yolo26n.pt')  # Модель людей (для каскада)
        self.conf_threshold = conf_threshold
        self.device = device
        self.cascade = cascade  # True = каскад, False = прямая детекция

    def set_cascade(self, enabled: bool):
        """Переключение метода детекции."""
        self.cascade = enabled

    def detect(self, frame):
        if self.cascade:
            return self._detect_cascade(frame)
        else:
            return self._detect_direct(frame)

    def _detect_direct(self, frame):
        """Прямая детекция """
        results = self.model(frame, conf=self.conf_threshold,
                             device=self.device, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in [0, 1]:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class_id': cls_id
                        })
        return detections

    def _detect_cascade(self, frame):
        """Каскадная детекция """
        # Этап 1: поиск людей
        person_results = self.person_model(
            frame, conf=self.conf_threshold,
            device=self.device, verbose=False, classes=[0]
        )

        # Сбор людей
        persons = []
        for result in person_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    persons.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(box.conf[0])
                    })

        # Этап 2: поиск касок
        helmet_results = self.model(frame, conf=self.conf_threshold,
                                    device=self.device, verbose=False)

        helmets = []
        no_helmets = []
        for result in helmet_results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 0:
                        helmets.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
                    elif cls == 1:
                        no_helmets.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})

        # Этап 3: сопоставление
        detections = []
        for person in persons:
            px1, py1, px2, py2 = person['bbox']

            has_helmet = False
            for helmet in helmets:
                hx1, hy1, hx2, hy2 = helmet['bbox']
                hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
                if px1 <= hcx <= px2 and py1 <= hcy <= py2:
                    has_helmet = True
                    break

            has_no_helmet = False
            for noh in no_helmets:
                nx1, ny1, nx2, ny2 = noh['bbox']
                ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
                if px1 <= ncx <= px2 and py1 <= ncy <= py2:
                    has_no_helmet = True
                    break

            if has_helmet:
                detections.append({
                    'bbox': person['bbox'],
                    'confidence': person['confidence'],
                    'class_id': 0
                })
            elif has_no_helmet:
                detections.append({
                    'bbox': person['bbox'],
                    'confidence': person['confidence'],
                    'class_id': 1
                })

        return detections