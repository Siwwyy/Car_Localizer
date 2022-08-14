import cv2


def draw_box(image, x: int, y: int, width: int, height: int):
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), thickness=1)
