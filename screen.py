"""화면 캡처와 게임 캔버스(파란 테두리) 자동 탐색."""
import mss
import numpy as np
from PIL import Image

BORDER = (29, 78, 216)  # 게임 캔버스 테두리 색


def grab_screen() -> Image.Image:
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def find_game(screen: Image.Image) -> tuple[int, int, int, int]:
    """파란 테두리 픽셀의 바깥 경계에서 캔버스 안쪽 영역 (left, top, right, bottom)을 구한다."""
    arr = np.asarray(screen)
    mask = (np.abs(arr.astype(int) - BORDER).sum(axis=2) < 30)
    # 테두리는 긴 직선이다. 다른 창이 일부 가려도 잡히도록 임계값을 낮게 둔다
    top, bottom = _border_pair(mask.sum(axis=1))
    left, right = _border_pair(mask.sum(axis=0))
    pad = 3  # 테두리 두께만큼 안쪽으로
    return int(left + pad), int(top + pad), int(right - pad + 1), int(bottom - pad + 1)


def _border_pair(counts: np.ndarray, min_count: int = 60, size: tuple = (200, 500)) -> tuple[int, int]:
    """파란 픽셀 수가 많은 줄 중 간격이 캔버스 크기와 맞는 (처음, 끝) 쌍을 찾는다."""
    idx = np.where(counts > min_count)[0]
    if len(idx) == 0:
        raise RuntimeError("게임 영역(파란 테두리)을 찾지 못했습니다. 게임 팝업이 화면에 보이는지 확인하세요.")
    lines = [int(i) for k, i in enumerate(idx) if k == 0 or i - idx[k - 1] > 1]  # 두께 2px 줄을 하나로 합침
    ends = [int(i) for k, i in enumerate(idx) if k == len(idx) - 1 or idx[k + 1] - i > 1]
    for a in lines:
        for b in reversed(ends):
            if size[0] < b - a < size[1]:
                return a, b
    raise RuntimeError(
        "게임 테두리가 일부만 보입니다. 터미널 등 다른 창이 게임 화면을 가리지 않게 옮기고 다시 실행하세요."
    )


def capture_game(region: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = region
    with mss.mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": right - left, "height": bottom - top})
    return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def has_motion(region: tuple[int, int, int, int], gap: float = 0.2) -> bool:
    """캔버스 위쪽 80%에서 물체가 움직이는지 본다. 바구니는 아래쪽이라 제외해 마우스 움직임에 속지 않는다."""
    import time

    a = np.asarray(capture_game(region).convert("L"), dtype=int)
    time.sleep(gap)
    b = np.asarray(capture_game(region).convert("L"), dtype=int)
    h = int(a.shape[0] * 0.8)
    return int((np.abs(a[:h] - b[:h]) > 40).sum()) > 40


def wait_for_round(should_stop, sustain: int = 3, interval: float = 0.1):
    """게임 시작(= 같은 팝업 위에서 물체가 연속으로 움직임)을 기다린다. 중단되면 None.

    팝업을 여닫는 애니메이션은 0.3초 안팎이라, has_motion(약 0.2초)이 sustain번 연속 True여야 시작으로 인정한다."""
    import time

    count, last = 0, None
    while not should_stop():
        try:
            region = find_game(grab_screen())
        except RuntimeError:
            count, last = 0, None
            time.sleep(interval)
            continue
        if region != last:  # 팝업이 새로 뜨거나 위치가 바뀌면 처음부터 센다
            count = 0
        last = region
        if has_motion(region):
            count += 1
            if count >= sustain:
                return region
        else:
            count = 0
    return None
