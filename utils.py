import pickle
from skimage.transform import resize
import numpy as np
import cv2
from PIL import Image

import torch
from model import Net

PATH = "model.pth"


EMPTY = True
NOT_EMPTY = False

# MODEL = pickle.load(open("model.pickle", "rb"))

net_load = Net()
net_load.load_state_dict(torch.load(PATH))

trans = pickle.load(open('transform.pickle', 'rb'))


def empty_or_not(spot_bgr):

    img = Image.fromarray(spot_bgr)
    transformed_img = trans(img)

    tensor = torch.unsqueeze(transformed_img, dim=0)

    out = net_load(tensor)

    _, predicted = torch.max(out.data, 1)

    y_output = predicted.item()


    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots
