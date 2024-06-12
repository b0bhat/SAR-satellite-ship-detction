import cv2
import sys
import numpy as np

# from deepdespeckling.utils.constants import PATCH_SIZE, STRIDE_SIZE
# from deepdespeckling.merlin.merlin_denoiser import MerlinDenoiser

image_path = sys.argv[1]

image = cv2.imread('data/images/' + image_path + '.jpg')

class_list = ['SHIP', 'SHIP', 'CAR', 'TANK', 'BRIDGE', 'HARBOR']
colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]

height, width, _ = image.shape

# i_npy = image.astype(np.float32)
# denoiser = MerlinDenoiser(model_name="spotlight", symetrise=True)
# denoised_image = denoiser.denoise_image(image, patch_size=PATCH_SIZE, stride_size=STRIDE_SIZE)
# image = denoised_image.astype(np.uint8)

with open('data/labels/' + image_path + '.txt', "r") as file1:
    for line in file1.readlines():
        split = line.split(" ")

        class_id = int(split[0])
        color = colors[class_id]
        clazz = class_list[class_id]

        x, y, w, h = float(split[1]), float(split[2]), float(split[3]), float(split[4])

        box = [int((x - 0.5*w)* width), int((y - 0.5*h) * height), int(w*width), int(h*height)]
        cv2.rectangle(image, box, color, 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,0,0))

cv2.imshow("output", image)
cv2.waitKey()