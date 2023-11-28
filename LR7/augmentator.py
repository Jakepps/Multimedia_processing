import csv
import pathlib
import cv2
import pytesseract
from pytesseract import Output
from difflib import SequenceMatcher
import statistics


def rel_path(rel_path):
    path = pathlib.Path(__file__).parent / rel_path
    return path


dataset_name = "dataset"
new_dataset_name = "dataset2"

img_files = list(pathlib.Path(str(rel_path(dataset_name))).glob("*.jpg"))
labels = {}
csv_str = ""

with open(
    str(rel_path(dataset_name + "/labels.csv")), newline="", encoding="utf-8"
) as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar="'")
    for row in reader:
        labels[row[0]] = row[1]


for img_file in img_files:
    img = cv2.imread(str(img_file.resolve()), 0)
    for angle in range(-20, 21, 1):
        height, width = img.shape[:2]  # image shape has 3 dimensions
        image_center = (
            width / 2,
            height / 2,
        )

        # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
        # Displaying the image
        new_file_name = (
            img_file.name[0 : len(img_file.name) - 4] + "_" + str(angle) + ".jpg"
        )
        output_str = new_dataset_name + "/" + new_file_name
        cv2.imwrite(
            output_str,
            rotated_mat,
        )
        csv_str += new_file_name + "," + labels[img_file.name] + "\n"

with open(new_dataset_name + "\labels.csv", "w", encoding="utf-8") as text_file:
    text_file.write(csv_str)
