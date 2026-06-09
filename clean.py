import os
from pathlib import Path

base_dir = Path("construction-ppe")

print("ВНИМАНИЕ: скрипт удалит файлы. Сделайте резервную копию!")
input("Нажмите Enter для продолжения или Ctrl+C для отмены...")

for split in ["train", "val", "test"]:
    img_dir = base_dir / "images" / split
    lbl_dir = base_dir / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        continue

    images = set()
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in img_dir.glob(ext):
            images.add(f.stem)

    labels = set()
    for f in lbl_dir.glob("*.txt"):
        labels.add(f.stem)

    # Удаляем изображения без разметки
    for stem in images - labels:
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = img_dir / f"{stem}{ext}"
            if img_path.exists():
                img_path.unlink()
                print(f"Удалено: {img_path}")

    # Удаляем разметку без изображений
    for stem in labels - images:
        lbl_path = lbl_dir / f"{stem}.txt"
        if lbl_path.exists():
            lbl_path.unlink()
            print(f"Удалено: {lbl_path}")

    # Удаляем пустые файлы разметки
    for f in lbl_dir.glob("*.txt"):
        if f.stat().st_size == 0:
            f.unlink()
            print(f"Удалён пустой: {f}")

print("\nОчистка завершена.")


