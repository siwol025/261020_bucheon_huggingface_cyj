"""로컬 블롭 검출: 단색 배경 위의 물체를 즉시(수 ms) 찾는다. 원격 DETR의 대안(--detector local)."""
import cv2
import numpy as np
from PIL import Image

from vision import Box

BG_MAX = 45          # 배경(13,17,23)보다 밝은 픽셀을 물체로 본다
MIN_AREA, MAX_AREA = 60, 3000
MIN_SIDE, MAX_SIDE = 10, 60


def detect(img: Image.Image) -> list[Box]:
    arr = np.asarray(img.convert("RGB"))
    mask = (arr.max(axis=2) > BG_MAX).astype(np.uint8)
    mask[:2, :] = mask[-2:, :] = 0  # 캔버스 테두리 잔여 픽셀 제거
    mask[:, :2] = mask[:, -2:] = 0
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # 폭탄 심지 같은 떨어진 조각을 합친다
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if MIN_AREA <= area <= MAX_AREA and MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE:
            boxes.append(Box(int(x) + 2, int(y) + 2, int(x + w) - 2, int(y + h) - 2, 1.0))  # dilate만큼 되돌림
    return boxes
