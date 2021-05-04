from pathlib import Path
import shutil

import cv2


dataset_name = "oidv6"
original_data_dir = Path.cwd().parent / dataset_name
converted_data_dir = original_data_dir.parent / f"{dataset_name}_converted"

classes = [
    "apple",
    "banana",
    "box",
    "cart",
    "gas_stove",
    "kitchen_&_dining_room_table",
    "orange",
    "table",
    "wheelchair",
    "wok",
    "wood-burning_stove",
]

for subset in ["train", "validation", "test"]:
    image_dir = converted_data_dir / "images" / subset
    label_dir = converted_data_dir / "labels" / subset
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, class_name in enumerate(classes):
        print(i, class_name)
        src_class_dir = original_data_dir / subset / class_name
        for filename in src_class_dir.glob("*.jpg"):
            shutil.copy(filename, image_dir / filename.name)
            label_filename = f"{filename.stem}.txt"
            curr_image = cv2.imread(str(filename))
            with open(src_class_dir / "labels" / label_filename, "r") as infile, open(
                label_dir / label_filename, "w"
            ) as outfile:
                for line in infile.readlines():
                    x_min, y_min, x_max, y_max = map(float, line.split(" ")[1:])
                    outfile.write(
                        f"{i} {x_min / curr_image.shape[1]} "
                        f"{y_min / curr_image.shape[0]} "
                        f"{(x_max - x_min) / curr_image.shape[1]} "
                        f"{(y_max - y_min) / curr_image.shape[0]}\n"
                    )
