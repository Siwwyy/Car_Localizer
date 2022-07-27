import cv2


def draw_box(image, x, y, width, height):
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), thickness=1)
