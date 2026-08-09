import time

from decoupled_wbc.control.real_safe.lowcmd_guard import NoCatchUpScheduler


def test_500hz_wall_clock_scheduler_has_no_catch_up_burst() -> None:
    starts = []
    errors = []

    def callback():
        starts.append(time.monotonic())
        if len(starts) % 40 == 0:
            time.sleep(0.006)

    scheduler = NoCatchUpScheduler(500.0, callback, on_error=errors.append)
    scheduler.start()
    time.sleep(1.0)
    scheduler.stop()

    intervals = [right - left for left, right in zip(starts, starts[1:])]
    assert len(intervals) > 300
    assert errors == []
    assert scheduler.metrics.missed_deadlines > 0
    assert min(intervals) > 0.001
    assert scheduler.metrics.summary()["p99_s"] < 0.010


def test_callback_error_stops_scheduler_without_retry_burst() -> None:
    calls = []
    errors = []

    def callback():
        calls.append(time.monotonic())
        if len(calls) == 5:
            raise RuntimeError("writer failed")

    scheduler = NoCatchUpScheduler(500.0, callback, on_error=errors.append)
    scheduler.start()
    time.sleep(0.05)
    scheduler.stop()
    assert len(calls) == 5
    assert len(errors) == 1
    assert scheduler.metrics.callback_errors == 1
