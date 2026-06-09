import os
from pathlib import Path

base_dir = Path("construction-ppe")

for split in ["train", "val", "test"]:
    img_dir = base_dir / "images" / split
    lbl_dir = base_dir / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"{split}: папка не найдена, пропускаю")
        continue

    # Все изображения
    images = set()
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for f in img_dir.glob(ext):
            images.add(f.stem)

    # Все файлы разметки
    labels = set()
    for f in lbl_dir.glob("*.txt"):
        labels.add(f.stem)

    # Изображения без разметки
    imgs_without_labels = images - labels
    # Разметка без изображений
    labels_without_imgs = labels - images
    # Общие
    matched = images & labels

    # Пустые labels (файл есть, но внутри пусто)
    empty_labels = []
    for f in lbl_dir.glob("*.txt"):
        if f.stat().st_size == 0:
            empty_labels.append(f.stem)

    print(f"\n{split}:")
    print(f"  Изображений:              {len(images)}")
    print(f"  Файлов разметки:          {len(labels)}")
    print(f"  Совпадает:                {len(matched)}")
    print(f"  Изображений БЕЗ разметки: {len(imgs_without_labels)}")
    print(f"  Разметки БЕЗ изображений: {len(labels_without_imgs)}")
    print(f"  Пустых файлов разметки:   {len(empty_labels)}")

print("\nДля удаления лишнего запусти clean_dataset.py")