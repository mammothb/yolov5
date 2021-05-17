import argparse
import functools
import multiprocessing
from pathlib import Path
import shutil

import cv2
import numpy as np


def convert(args):
    original_data_dir = Path.cwd().parent / args.name
    converted_data_dir = original_data_dir.parent / "_".join(
        [args.name, "converted"] + args.modifier.split(",")
    )

    # classes = [
    #     "apple",
    #     "banana",
    #     "box",
    #     "cart",
    #     "gas_stove",
    #     "kitchen_&_dining_room_table",
    #     "orange",
    #     "table",
    #     "wheelchair",
    #     "wok",
    #     "wood-burning_stove",
    # ]
    # classes = ["apple", "banana", "orange", "wheelchair", "wok"]
    # classes = ["box"]
    classes = ["table"]

    for subset in ["train", "test"]:
        print(subset)

        image_dir = converted_data_dir / "images" / subset
        label_dir = converted_data_dir / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        src_data_dir = original_data_dir / "multidata" / subset
        convert_file_partial = functools.partial(
            convert_file,
            classes=classes,
            image_dir=image_dir,
            label_dir=label_dir,
            src_data_dir=src_data_dir,
        )
        with multiprocessing.Pool(args.j) as p:
            p.map(convert_file_partial, src_data_dir.glob("*.jpg"))


def convert_file(filename, classes, image_dir, label_dir, src_data_dir):
    shutil.copy(filename, image_dir / filename.name)
    label_filename = f"{filename.stem}.txt"
    curr_image = cv2.imread(str(filename))
    with open(src_data_dir / "labels" / label_filename, "r") as infile, open(
        label_dir / label_filename, "w"
    ) as outfile:
        l = [x.split() for x in infile.read().strip().splitlines()]
        for line_parts in np.unique(l, axis=0):
            x_min, y_min, x_max, y_max = map(float, line_parts[1:])
            outfile.write(
                f"{classes.index(line_parts[0])} "
                f"{(x_min + x_max) / 2 / curr_image.shape[1]} "
                f"{(y_min + y_max) / 2 / curr_image.shape[0]} "
                f"{(x_max - x_min) / curr_image.shape[1]} "
                f"{(y_max - y_min) / curr_image.shape[0]}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="oidv6_data", help="original data path"
    )
    parser.add_argument("--modifier", type=str, default="", help="dataset modifiers")
    parser.add_argument("--j", type=int, default=1, help="job number")
    args = parser.parse_args()
    convert(args)
