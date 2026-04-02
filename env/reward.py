# ── env/reward.py ─────────────────────────────────────────────────────────────
# Two reward modes, selected by config.REWARD_MODE.
#
# checkpoint → +1 per new waypoint passed, -10 if off track.
# laptime    → progress * speed * dt, -10 if off track.

from config import (REWARD_MODE, CHECKPOINT_REWARD,
                    OFF_TRACK_PENALTY, SPEED_WEIGHT, STEP_PENALTY, DT)


def compute_reward(
    on_track: bool,
    new_waypoints: int,
    speed: float,
    prev_progress: float,
    curr_progress: float,
) -> float:
    if not on_track:
        return OFF_TRACK_PENALTY

    reward = -STEP_PENALTY

    if REWARD_MODE == "checkpoint":
        return reward + float(new_waypoints) * CHECKPOINT_REWARD

    elif REWARD_MODE == "laptime":
        progress_delta = curr_progress - prev_progress
        # Handle wrap-around at lap completion
        if progress_delta < 0:
            # progress_delta stays negative only if agent went backward;
            # a lap completion gives a large positive jump — clamp negatives to 0
            progress_delta = 0.0
        return reward + (progress_delta * speed * SPEED_WEIGHT)

    return reward