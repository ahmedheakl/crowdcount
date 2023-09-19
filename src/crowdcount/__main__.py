import os
import sys

import numpy as np
import cv2

from crowdcount.crowd_count import CrowdCounter
from crowdcount import network


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = int((ht / 4) * 4)
    wd_1 = int((wd / 4) * 4)
    img = cv2.resize(img, (wd_1, ht_1))
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))
    return img


def postprocess(density_map):
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    result_img = density_map.astype(np.uint8, copy=False)
    result_img = cv2.applyColorMap(result_img, cv2.COLORMAP_JET)
    return result_img


def main(option: str):
    model_path = "cmtl_shtechB_768.h5"
    net = CrowdCounter()

    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.eval()
    if option == "img":
        img_count(net)
    elif option == "vid":
        vid_count(net)
    else:
        print("Invalid option")


def img_count(net):
    img_path = (
        "/home/heakl/Desktop/crowd-counting/crowdcount-cascaded-mtl/shopping_mall.jpg"
    )
    img = cv2.imread(img_path)
    im_data = preprocess(img)
    density_map = net(im_data)
    density_map = density_map.detach().numpy()
    et_count = np.sum(density_map)
    print(et_count)


def vid_count(net):
    video_path = (
        "/home/heakl/Desktop/crowd-counting/crowdcount-cascaded-mtl/video_sample.mp4"
    )
    cap = cv2.VideoCapture(video_path)

    while True:
        _, frame = cap.read()
        im_data = preprocess(frame)
        density_map = net(im_data)
        density_map = density_map.detach().numpy()
        et_count = np.sum(density_map)
        print(et_count)
        result_img = postprocess(density_map)
        cv2.imshow("Result", result_img)
        # cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1])
