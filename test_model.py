from ultralytics import YOLO
from datetime import datetime

if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()

    models = [
        "best_yolo26n.pt",
        "best_yolo26s.pt",
        "best_yolo26m.pt",
        "best_yolo26n_aug.pt",
        "best_yolo26s_aug.pt",
        "best_yolo26m_aug.pt",
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Результаты тестирования моделей ({datetime.now().strftime('%d.%m.%Y %H:%M:%S')})\n")
        f.write("=" * 67 + "\n")
        f.write(f"{'Модель':<25} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 67 + "\n")

        for model_path in models:
            try:
                print(f"\n{'=' * 50}")
                print(f"Модель: {model_path}")
                print('=' * 50)

                m = YOLO(model_path)
                r = m.val(data='construction-ppe/helmet_only.yaml', split='test', device=0, verbose=False)

                map50 = r.box.map50
                map50_95 = r.box.map
                p = r.box.p[0]
                rec = r.box.r[0]

                print(f"mAP@0.5:      {map50:.4f}")
                print(f"mAP@0.5:0.95: {map50_95:.4f}")
                print(f"Precision:    {p:.4f}")
                print(f"Recall:       {rec:.4f}")

                f.write(f"{model_path:<25} {map50:<10.4f} {map50_95:<12.4f} {p:<10.4f} {rec:<10.4f}\n")

                choice = input("\nEnter — следующая модель, Q — выход: ").strip().lower()
                if choice == 'q':
                    print("Выход.")
                    break
            except Exception as e:
                print(f"Ошибка: {e}")
                f.write(f"{model_path:<25} {'ERROR':<10}\n")
                choice = input("\nEnter — продолжить, Q — выход: ").strip().lower()
                if choice == 'q':
                    print("Выход.")
                    break

        f.write("-" * 67 + "\n")
        f.write("Тестирование завершено.\n")

    print(f"\nРезультаты сохранены в: {filename}")