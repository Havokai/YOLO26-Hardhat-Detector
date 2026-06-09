import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo26m.pt")

    results = model.train(
        data="construction-ppe/helmet_only.yaml",
        epochs=100,
        imgsz=640,
        batch=12,
        name="helmet_final_yolo26",
        device=0,
        workers=8,
        lr0=0.001,
        lrf=0.01,
        patience=10,
        cos_lr=True,
        cls=1.0,
        dropout=0.1,
        amp=True,

        # Аугментация
        hsv_h=0.018,
        hsv_s=0.65,
        hsv_v=0.35,
        degrees=20,
        translate=0.15,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,

        # Дополнительная аугментация
        # copy_paste=0.1,
        # erasing=0.4,
        # perspective=0.0005,
    )

    print("\nСохранение модели...")
    model.save("best_yolo26m_aug.pt")

    metrics = model.val()
    print(f"\nИТОГОВЫЕ МЕТРИКИ:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p[0]:.4f}" if metrics.box.p is not None else "Precision: N/A")
    print(f"Recall: {metrics.box.r[0]:.4f}" if metrics.box.r is not None else "Recall: N/A")
    if metrics.box.p is not None and metrics.box.r is not None:
        p = metrics.box.p[0]
        r = metrics.box.r[0]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"F1-score: {f1:.4f}")