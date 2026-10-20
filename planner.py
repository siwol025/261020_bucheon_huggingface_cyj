"""낙하물 추적과 이동 판단. 네트워크/화면 의존성이 없어 단독 테스트가 가능하다."""
from dataclasses import dataclass, field

DEFAULT_VY = 130.0  # 속도를 측정하기 전에 쓰는 낙하 속도(px/s)
MATCH_DX = 30.0     # 같은 물체로 볼 최대 x 차이(px). 낙하물은 x가 거의 일정하다
MATCH_DY = 280.0    # 같은 물체로 볼 최대 y 이동량(px). 인식 주기(~1초) 동안 내려간 거리
STALE_SEC = 2.5     # 이 시간 동안 관측되지 않으면 추적 종료
LEAD_SEC = 1.0      # 도착 이 시간 전부터 음식 쪽으로 이동
BOMB_HALF = 32      # 폭탄과 바구니가 겹친다고 보는 x 거리(px)
BOMB_WINDOW = 0.35  # 폭탄 도착 시각이 이 범위 안이면 위험(초)


VY_WINDOW = 1.0       # 속도 추정에 쓰는 최근 관측 구간(초)
VY_MIN_SPAN = 0.3     # 이 시간 이상 관측해야 속도를 믿는다
GRACE_SEC = 0.10      # 떨어질 시각이 지나도 이 시간까지는 대상으로 유지(지연 보정)


@dataclass
class Track:
    label: str
    x: float
    y: float
    t: float
    vy: float = DEFAULT_VY
    seen: int = 1
    measured: bool = False
    history: list = field(default_factory=list)  # 최근 (t, y) 관측

    def __post_init__(self):
        self.history = [(self.t, self.y)]

    def y_at(self, now: float) -> float:
        return self.y + self.vy * (now - self.t)

    def update(self, x: float, y: float, t: float) -> bool:
        """관측을 반영한다. 처음으로 속도가 측정되면 True."""
        self.history = [(ht, hy) for ht, hy in self.history if t - ht <= VY_WINDOW] + [(t, y)]
        self.x, self.y, self.t = x, y, t
        self.seen += 1
        if len(self.history) < 2 or t - self.history[0][0] < VY_MIN_SPAN:
            return False
        ts = [h[0] for h in self.history]
        ys = [h[1] for h in self.history]
        tm, ym = sum(ts) / len(ts), sum(ys) / len(ys)
        var = sum((a - tm) ** 2 for a in ts)
        vy = sum((a - tm) * (b - ym) for a, b in zip(ts, ys)) / var  # 최소제곱 기울기
        if not 20 < vy < 800:  # 비정상 값은 무시
            return False
        first = not self.measured
        self.vy, self.measured = vy, True
        return first


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.default_vy = DEFAULT_VY

    def match(self, boxes, now: float):
        """탐지 박스 [(cx, cy)]를 기존 추적과 연결하고, 연결 안 된 박스의 인덱스를 반환한다."""
        used = set()
        unmatched = []
        for i, (cx, cy) in enumerate(boxes):
            best, best_dy = None, MATCH_DY
            for tr in self.tracks:
                if id(tr) in used or abs(tr.x - cx) > MATCH_DX:
                    continue
                dy = cy - tr.y  # 마지막 관측 이후 아래로 내려온 거리
                if -15 < dy < best_dy:
                    best, best_dy = tr, dy
            if best is None:
                unmatched.append(i)
            else:
                used.add(id(best))
                if best.update(cx, cy, now):  # 처음 측정된 속도는 새 물체의 기본 속도에 반영한다
                    self.default_vy = 0.7 * self.default_vy + 0.3 * best.vy
        return unmatched

    def add(self, label: str, x: float, y: float, t: float):
        tr = Track(label, x, y, t, self.default_vy)
        self.tracks.append(tr)
        return tr

    def prune(self, now: float, bottom: float):
        self.tracks = [
            tr for tr in self.tracks
            if now - tr.t < STALE_SEC and tr.y_at(now) < bottom + 20
        ]


def choose_target(tracks, now: float, catch_y: float, width: float, current_x: float):
    """바구니가 가야 할 x를 반환한다. 움직일 필요가 없으면 None."""
    def tta(tr):  # time to arrive: 바구니 높이에 닿기까지 남은 시간
        return (catch_y - tr.y_at(now)) / max(tr.vy, 1.0)

    bombs = [(tr.x, tta(tr)) for tr in tracks if tr.label == "BOMB"]

    def dangerous(x, t_from, t_to):
        return any(abs(bx - x) < BOMB_HALF and t_from < bt < t_to for bx, bt in bombs)

    foods = sorted(
        ((tta(tr), tr) for tr in tracks if tr.label == "FOOD" and tta(tr) > -GRACE_SEC),
        key=lambda p: p[0],
    )
    for t_arr, tr in foods:
        if dangerous(tr.x, t_arr - BOMB_WINDOW, t_arr + BOMB_WINDOW):
            continue  # 같은 자리에 폭탄이 같이 떨어지면 포기
        if t_arr <= LEAD_SEC:
            return tr.x
        break  # 가장 가까운 음식이 아직 멀면 제자리 유지

    # 현재 자리로 폭탄이 오면 폭탄에서 가장 먼 x로 피한다
    if dangerous(current_x, -BOMB_WINDOW, LEAD_SEC):
        candidates = range(20, int(width) - 20, 10)
        return max(
            candidates,
            key=lambda x: min([abs(x - bx) for bx, _ in bombs] or [width]),
        )
    return None
