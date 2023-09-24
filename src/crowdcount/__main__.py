"""Main script for crowd counting"""
from typing import Tuple
import argparse

import numpy as np
import cv2

from crowdcount.crowd_count import CrowdCounter
from crowdcount import network
from crowdcount.utils import preprocess, generate_density_map, download_file
from crowdcount import WEIGHTS_LINK


def predict(net: CrowdCounter, img: np.ndarray) -> Tuple[int, np.ndarray]:
    """Run prediction on image

    Args:
        img (np.ndarray): Input image

    Returns:
        Tuple[int, np.ndarray]: [Estimated count, Density Map]
    """
    im_data = preprocess(img)
    density_map = net(im_data)
    density_map = density_map.detach().numpy()
    et_count = np.sum(density_map)
    return et_count, density_map


def predict_video(net, video_path: str) -> None:
    """Run prediction on video

    Args:
        net (_type_): Network to use for prediction
        video_path (str): Video index or path
    """
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() is False:
        raise ValueError("Video path is invalid")

    while True:
        _, frame = cap.read()
        et_count, density_map = predict(net, frame)
        print(et_count)
        result_img = generate_density_map(density_map)
        cv2.imshow("Result", result_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_img(net: CrowdCounter, img_path: str) -> Tuple[int, np.ndarray]:
    """Run inferece on input image

    Args:
        net (CrowdCounter): CrowdCounter instance
        img_path (str): Path to image

    Returns:
        Tuple[int, np.ndarray]: [Estimated Count, Density Map]
    """
    img = cv2.imread(img_path)
    im_data = preprocess(img)
    density_map = net(im_data)
    density_map = density_map.detach().numpy()
    et_count = np.sum(density_map)
    print(et_count)
    return et_count, density_map


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="Crowd Counter made easy")
    parser.add_argument(
        "--mode",
        "-m",
        default="video",
        type=str,
        choices=["video", "image"],
        help="Type of inference whether video or image",
    )
    parser.add_argument(
        "--path",
        "-p",
        default=None,
        type=str,
        help="Path to video or Image",
    )
    args = parser.parse_args()
    if args.mode == "image" and args.path is None:
        raise ValueError("Image path must be set")

    if args.path is None:
        args.path = 0
    model_path = download_file(WEIGHTS_LINK, "cmtl_shtechB_768.h5")
    print("PATH TO THE MODEL", model_path)
    net = CrowdCounter()
    network.load_net(model_path, net)
    net.eval()
    if args.mode == "image":
        predict_img(net, args.path)
    else:
        predict_video(net, args.path)


if __name__ == "__main__":
    main()
