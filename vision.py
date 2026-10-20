"""HuggingFace 원격 추론 호출 모음: DETR 위치 탐지 + Gemma 3 종류 분류."""
import base64
import io
import os
import tempfile
import uuid
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient
import numpy as np
from PIL import Image

load_dotenv(find_dotenv())

DETR_MODEL = "facebook/detr-resnet-50"
VLM_MODEL = "google/gemma-3-27b-it"
MIN_SCORE = 0.35

_detr = InferenceClient(provider="hf-inference", api_key=os.environ["HF_TOKEN"])
_vlm = InferenceClient(api_key=os.environ["HF_TOKEN"])


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


def detect(img: Image.Image) -> list[Box]:
    """DETR로 물체 박스를 찾는다. 라벨(COCO)은 게임 아이콘과 맞지 않아 버리고 위치만 쓴다."""
    # InferenceClient는 PIL 객체/bytes를 받지 못해 임시 파일 경로로 넘긴다
    path = os.path.join(tempfile.gettempdir(), f"detr_{uuid.uuid4().hex}.png")
    img.save(path)
    try:
        out = _detr.object_detection(path, model=DETR_MODEL)
    finally:
        os.remove(path)
    return [
        Box(o.box.xmin, o.box.ymin, o.box.xmax, o.box.ymax, o.score)
        for o in out
        if o.score >= MIN_SCORE
    ]


def classify(crop: Image.Image) -> str:
    """Gemma 3에게 크롭 이미지를 보여주고 FOOD / BOMB 중 하나를 받는다."""
    crop = crop.resize((128, 128))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    res = _vlm.chat.completions.create(
        model=VLM_MODEL,
        max_tokens=5,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Is this a BOMB or a FOOD item? Answer with one word: BOMB or FOOD."},
                ],
            }
        ],
    )
    word = res.choices[0].message.content.strip().upper()
    for label in ("BOMB", "FOOD", "BASKET"):
        if label in word:
            return label
    return "OTHER"


def quick_classify(crop: Image.Image) -> str:
    """색으로 즉시(수 ms) 임시 판정한다. 폭탄은 어두운 회색, 음식은 채도가 높다.
    Gemma 분류가 끝나기 전까지 쓰고, 결과가 다르면 Gemma가 정정한다."""
    hsv = np.asarray(crop.convert("HSV")).reshape(-1, 3).astype(float)
    rgb = np.asarray(crop.convert("RGB")).reshape(-1, 3).astype(float)
    fg = rgb.max(axis=1) > 40  # 배경(거의 검정)보다 밝은 픽셀만
    if fg.sum() < 20:
        return "OTHER"
    return "FOOD" if (hsv[fg, 1] / 255 > 0.4).mean() > 0.4 else "BOMB"
