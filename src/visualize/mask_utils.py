import cv2
import numpy as np


def draw_mask_contours(masks: np.ndarray, origin_img: np.ndarray, colors: list[tuple], line_type=cv2.LINE_AA):
    """
    draw contours of masks on the original image.
    shape of masks does not have to match the shape of the original image.
    masks is resized to the shape of the original image.

    Parameters
    ----------
    masks: np.ndarray
        masks of shape (n_classes, height, width)
    origin_img: np.ndarray
        original image of shape (height, width, 3)
    colors: list[tuple]
        list of color codes (ex. [(255, 0, 0), (0, 255, 0)])
        each color is based on BGR
    line_type: int
        type of line to draw contours

    Returns
    -------
    img_contour: np.ndarray
        image with contours
    """

    assert masks.ndim == 3, "masks should be of shape (n_classes, height, width)"

    origin = origin_img.copy()
    origin_h, origin_w = origin.shape[0], origin.shape[1]

    num_classes = masks.shape[0]
    img_contour = None
    for i in range(num_classes):
        mask = masks[i]  # (h, w)
        mask = np.expand_dims(mask, axis=2)  # (h, w) -> (h, w, 1)
        mask = cv2.resize(mask, (origin_w, origin_h))  # (h, w, 1) -> (origin_height, origin_width)
        mask = np.where(mask > 0, 255, 0)  # 0~1のfloatを 0 or 255 に変換（これをしておかないとあとで描画する輪郭がガビガビになる）
        mask = mask.astype(np.uint8)

        color = colors[i]

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 輪郭だけを描画
        img_contour = cv2.drawContours(image=origin, contours=contours, contourIdx=-1, color=color, thickness=2,
                                       lineType=line_type)

    return img_contour


def draw_batch_mask_contours(masks: np.ndarray, original_images: list[np.ndarray], colors: list[tuple],
                             line_type=cv2.LINE_AA):
    """
    draw contours of masks on the original image

    Parameters
    ----------
    masks: np.ndarray
        masks of shape (n_batches, n_classes, height, width)
    original_images: list[np.ndarray]
        original image of shape (n_batches, height, width, 3)
    colors: list[tuple]
        list of color codes (ex. [(255, 0, 0), (0, 255, 0)])
    line_type: int
        type of line to draw contours

    Returns
    -------
    img_contour: np.ndarray
        images with contours
    """

    assert masks.ndim == 4, "masks should be of shape (n_batches, n_classes, height, width)"
    n_batches = masks.shape[0]
    img_contours = []
    for i in range(n_batches):
        img_contour = draw_mask_contours(masks[i], original_images[i], colors, line_type)
        img_contours.append(img_contour)

    return img_contours
