"""먹이 받기 게임 자동 플레이 봇.

인식 스레드: 화면 캡처 → DETR 탐지 → 새 물체만 Gemma로 분류 (느리다, 약 1~2초 주기)
제어 루프  : 마지막 관측으로 현재 위치를 예측해 마우스로 바구니를 이동 (빠르다, 약 30Hz)

사용법: python bot.py [--seconds 30] [--dry] [--detector local|detr]
  --dry : 마우스를 움직이지 않고 판단만 출력
종료   : Ctrl+C
"""
import argparse
import sys
import ctypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyautogui
from PIL import Image

import planner
import screen
import local_detect
import vision

FREEZE_SEC = 2.0    # 이 시간 동안 화면이 정지해 있으면 게임이 끝난 것으로 본다
BASKET_ZONE = 0.80  # 캔버스 높이의 이 비율보다 아래 박스는 바구니/받은 음식이라 무시
CATCH_RATIO = 0.85  # 음식 중심이 바구니 윗면에 닿는 높이(캔버스 높이 비율, 실측)


def perception(region, tracker, lock, stop, pool, detect, activity):
    width, height = region[2] - region[0], region[3] - region[1]
    local = detect is local_detect.detect

    def crop_of(game, b, pad=6):
        return game.crop((max(int(b.x1) - pad, 0), max(int(b.y1) - pad, 0), int(b.x2) + pad, int(b.y2) + pad))

    def clipped(b):
        return b.y1 <= 5 or (b.y2 - b.y1) < 20

    def verify_async(track, crop, quick):
        """Gemma가 임시(색) 라벨을 확인하고, 다르면 정정한다."""
        label = vision.classify(crop)
        with lock:
            if label in ("FOOD", "BOMB"):
                track.label = label
        print(f"[Gemma] {quick} -> {label}{'' if label == quick else '  (정정)'}")

    def report_error(fut):  # 스레드풀은 예외를 삼키므로 직접 출력한다
        if fut.exception():
            print(f"[Gemma 오류] {type(fut.exception()).__name__}: {fut.exception()}")

    prev = None
    while not stop.is_set():
        t0 = time.time()
        try:
            game = screen.capture_game(region)
            gray = np.asarray(game.convert("L"), dtype=int)[: int(height * BASKET_ZONE)]
            if prev is not None and (np.abs(gray - prev) > 40).sum() > 40:
                activity["t"] = t0  # 화면이 움직이는 중
            prev = gray
            boxes = [b for b in detect(game) if b.cy < height * BASKET_ZONE]
            with lock:
                unmatched = tracker.match([(b.cx, b.cy) for b in boxes], t0)
                for i in unmatched:  # 색으로 즉시 임시 라벨을 붙이고, Gemma가 뒤에서 확인한다
                    if clipped(boxes[i]):
                        continue  # 위에서 막 나타나 일부만 보이면 색이 왜곡되므로(폭탄 심지 등) 다 보일 때까지 미룬다
                    crop = crop_of(game, boxes[i])
                    quick = vision.quick_classify(crop)
                    if quick == "OTHER":
                        continue
                    tr = tracker.add(quick, boxes[i].cx, boxes[i].cy, t0)
                    print(f"[발견] {quick} x={boxes[i].cx:.0f} y={boxes[i].cy:.0f} (속도 기본값 {tracker.default_vy:.0f}px/s)")
                    pool.submit(verify_async, tr, crop, quick).add_done_callback(report_error)
                tracker.prune(time.time(), height)
            if not local:  # 원격 탐지는 사이클마다 지연을 보여 준다
                print(f"[인식] {len(boxes)}개 탐지, 새 물체 {len(unmatched)}개, {time.time() - t0:.2f}s")
        except Exception as e:  # 네트워크 오류 한 번으로 봇이 멈추지 않게 한다
            print(f"[인식 오류] {type(e).__name__}: {e}")
            time.sleep(0.5)
        if local:
            time.sleep(0.02)


def warm_up():
    """첫 호출은 모델이 깨어나느라 수 초~10초 걸린다. 팝업이 뜨기 전에 미리 겪는다."""
    t = time.time()
    blank = Image.new("RGB", (294, 294), (13, 17, 23))
    vision.detect(blank)
    vision.classify(blank.crop((0, 0, 40, 40)))
    print(f"워밍업 완료 ({time.time() - t:.1f}s)")


class Tee:
    """print를 콘솔과 run_log.txt에 함께 기록한다. 줄마다 게임 시작 후 경과 시간을 붙인다."""

    def __init__(self, path):
        self.console, self.file, self.t0, self.buf = sys.stdout, open(path, "w", encoding="utf-8"), time.time(), ""

    def write(self, s):
        self.console.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.file.write(f"{time.time() - self.t0:6.2f}s {line}\n")
            self.file.flush()

    def flush(self):
        self.console.flush()


def listen_for_quit(quit_flag):
    """터미널에서 q를 누르면 종료 플래그를 세운다."""
    import msvcrt

    def run():
        while not quit_flag.is_set():
            if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                quit_flag.set()
            time.sleep(0.05)

    threading.Thread(target=run, daemon=True).start()


def play_round(region, pool, detect, args, quit_flag):
    """한 판을 플레이하고 마우스 이동 횟수를 반환한다."""
    left, top, right, bottom = region
    width, height = right - left, bottom - top
    catch_y = height * CATCH_RATIO

    tracker = planner.Tracker()
    lock = threading.Lock()
    stop = threading.Event()
    started = time.time()
    activity = {"t": started}  # 마지막으로 화면이 움직인 시각
    worker = threading.Thread(target=perception, args=(region, tracker, lock, stop, pool, detect, activity), daemon=True)
    worker.start()
    local = detect is local_detect.detect  # 원격 탐지는 프레임 간격이 불규칙해 정지 판정을 쓰지 않는다

    current_x = width / 2
    mouse_y = int(top + catch_y)
    end = started + args.seconds  # 안전 상한. 보통은 화면 정지로 먼저 끝난다
    last_report, moves = 0.0, 0
    try:
        while time.time() < end and not quit_flag.is_set():
            now = time.time()
            if local and now - started > 3 and now - activity["t"] > FREEZE_SEC:
                print("[종료 감지] 화면이 멈췄습니다 (게임 끝)")
                break
            with lock:
                target = planner.choose_target(list(tracker.tracks), now, catch_y, width, current_x)
            if target is not None and abs(target - current_x) > 3:
                current_x = max(10, min(width - 10, target))
                moves += 1
                print(f"[이동] x={current_x:.0f}{' (dry)' if args.dry else ''}")
                if not args.dry:
                    pyautogui.moveTo(left + current_x, mouse_y)
            if now - last_report > 3:
                last_report = now
                with lock:
                    speeds = [f"{t.label}@x{t.x:.0f}:{t.vy:.0f}" for t in tracker.tracks if t.measured]
                print(f"[속도] 기본 {tracker.default_vy:.0f}px/s, 측정된 물체 {speeds}")
            time.sleep(0.03)
    finally:
        stop.set()
        worker.join(timeout=2)
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=45, help="한 판 최대 시간(안전 상한). 보통은 화면이 멈추면 먼저 끝난다")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--detector", choices=["local", "detr"], default="local",
                    help="local: 로컬 블롭 검출(빠름, 기본) / detr: HF 원격 DETR(느림)")
    args = ap.parse_args()

    sys.stdout = Tee("run_log.txt")
    ctypes.windll.user32.SetProcessDPIAware()  # mss 픽셀 좌표와 pyautogui 좌표를 일치시킨다
    pyautogui.PAUSE = 0

    print("워밍업 중...")
    warm_up()
    detect = local_detect.detect if args.detector == "local" else vision.detect
    pool = ThreadPoolExecutor(max_workers=4)
    quit_flag = threading.Event()
    listen_for_quit(quit_flag)

    round_no = 0
    print(f"준비 완료! 게임 '플레이'를 누르세요. 몇 판이든 자동으로 이어서 플레이합니다. 종료: q 키 (탐지기={args.detector}{', dry' if args.dry else ''})")
    try:
        while not quit_flag.is_set():
            region = screen.wait_for_round(quit_flag.is_set)
            if region is None:
                break
            round_no += 1
            print(f"=== {round_no}번째 판 시작, 게임 영역 {region} ===")
            moves = play_round(region, pool, detect, args, quit_flag)
            print(f"=== {round_no}번째 판 종료 (이동 {moves}회) ===")
            if quit_flag.is_set():
                break
            print("다음 판을 기다립니다. 게임을 다시 시작하세요. 종료: q 키")
    except KeyboardInterrupt:
        pass
    print(f"종료 (총 {round_no}판)")


if __name__ == "__main__":
    main()
