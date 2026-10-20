"""planner.py 단위 테스트 (네트워크/화면 불필요). 실행: python test_planner.py"""
import planner as p

W, CATCH = 294, 258.0
now = 100.0


def tr(label, x, y, vy=120.0):
    t = p.Track(label, x, y, now, vy)
    return t


def test_goes_to_near_food():
    assert p.choose_target([tr("FOOD", 200, 200)], now, CATCH, W, 100) == 200


def test_waits_when_food_far():
    assert p.choose_target([tr("FOOD", 200, 20)], now, CATCH, W, 100) is None


def test_picks_lowest_food_first():
    t = p.choose_target([tr("FOOD", 50, 150), tr("FOOD", 250, 210)], now, CATCH, W, 100)
    assert t == 250


def test_skips_food_with_bomb_beside_it():
    t = p.choose_target([tr("FOOD", 150, 200), tr("BOMB", 155, 198), tr("FOOD", 60, 170)], now, CATCH, W, 100)
    assert t == 60


def test_dodges_bomb_on_current_spot():
    t = p.choose_target([tr("BOMB", 100, 200)], now, CATCH, W, 100)
    assert t is not None and abs(t - 100) > p.BOMB_HALF


def test_ignores_bomb_far_from_spot():
    assert p.choose_target([tr("BOMB", 250, 200)], now, CATCH, W, 100) is None


def test_velocity_estimate_and_prediction():
    tk = p.Tracker()
    tk.add("FOOD", 100, 50, 0.0)
    assert tk.match([(101, 110)], 0.5) == []  # 같은 물체로 연결
    assert abs(tk.tracks[0].vy - 120) < 1
    assert abs(tk.tracks[0].y_at(1.0) - 170) < 1


def test_new_object_is_unmatched():
    tk = p.Tracker()
    tk.add("FOOD", 100, 50, 0.0)
    assert tk.match([(250, 40)], 0.5) == [0]


def test_add_returns_track_so_label_can_be_set_later():
    tk = p.Tracker()
    track = tk.add("?", 100, 50, 0.0)
    track.label = "BOMB"
    assert tk.tracks[0].label == "BOMB"


def test_matches_fast_falling_object_between_slow_detections():
    tk = p.Tracker()
    tk.add("FOOD", 100, 40, 0.0)
    assert tk.match([(104, 190)], 1.0) == []  # 1초에 150px 낙하
    assert len(tk.tracks) == 1 and abs(tk.tracks[0].vy - 150) < 1


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
