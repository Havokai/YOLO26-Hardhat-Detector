import os
from pathlib import Path

def count_labels(labels_dir):
    """
    Подсчитывает количество объектов каждого класса во всех .txt файлах разметки.
    Возвращает словарь {class_id: count} и общее количество.
    """
    counts = {}
    total_files = 0
    total_objects = 0

    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue
        total_files += 1
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                counts[cls_id] = counts.get(cls_id, 0) + 1
                total_objects += 1

    return counts, total_files, total_objects


def main():
    base_dir = Path("construction-ppe/labels")

    for split in ["train", "val", "test"]:
        labels_dir = base_dir / split
        if not labels_dir.exists():
            print(f"{split}: папка не найдена")
            continue

        counts, files, objects = count_labels(labels_dir)

        print(f"\n{split}:")
        print(f"  Файлов разметки: {files}")
        print(f"  Всего объектов:  {objects}")
        for cls_id in sorted(counts.keys()):
            class_name = {0: "helmet", 1: "no_helmet"}.get(cls_id, f"class_{cls_id}")
            print(f"    {class_name} ({cls_id}): {counts[cls_id]}")

if __name__ == '__main__':
    main()