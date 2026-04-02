# ── env/car.py ────────────────────────────────────────────────────────────────
# Bicycle kinematic model + 7-ray sensor suite.
#
# State:  [x, y, heading (rad), speed (m/s)]
# Action: [steering (-1..1), acceleration (-1..1)]

import numpy as np
from config import (WHEELBASE, MAX_STEER, MAX_ACCEL, MAX_BRAKE,
                    MAX_SPEED, FRICTION, DRAG, CORNERING_G, TURN_FRICTION, DT, SENSOR_ANGLES, SENSOR_MAX_DIST, N_SENSORS)


class Car:
    def __init__(self, start_x: float, start_y: float, start_heading: float):
        self.reset(start_x, start_y, start_heading)

    def reset(self, x: float, y: float, heading: float):
        self.x       = float(x)
        self.y       = float(y)
        self.heading = float(heading)
        self.speed   = 10.0  # Rolling start — gives the agent immediate forward momentum

    # ── Step 3a: bicycle model step ───────────────────────────────────────────
    def step(self, steer: float, accel: float):
        """
        steer, accel both in [-1, 1].
        Positive accel = throttle, negative = brake.
        """
        # Map to physical units
        steer_angle = float(np.clip(steer, -1, 1)) * MAX_STEER

        # Grip Limit / Understeer: Limit maximum steering angle based on speed
        # Maximum lateral acceleration = v^2 / R <= MAX_LATERAL_G
        # R = WHEELBASE / tan(steer)  =>  tan(steer) <= (MAX_LATERAL_G * WHEELBASE) / v^2
        if self.speed > 1.0:
            max_tan = (CORNERING_G * 9.81 * WHEELBASE) / (self.speed ** 2)
            if np.abs(np.tan(steer_angle)) > max_tan:
                steer_angle = float(np.sign(steer_angle) * np.arctan(max_tan))

        if accel > 1e-4:
            delta_v =  float(np.clip(accel, 0, 1)) * MAX_ACCEL * DT
        elif accel < -1e-4:
            delta_v =  float(np.clip(accel, -1, 0)) * MAX_BRAKE * DT
        else:
            delta_v = 0.0

        # Apply mechanical friction and aerodynamic drag to mimic real speed decay
        friction = FRICTION * DT
        drag = DRAG * (self.speed ** 2) * DT
        
        # Turning Friction: Turning the wheels scrubs off speed
        turn_drag = TURN_FRICTION * abs(steer_angle) * (self.speed / MAX_SPEED) * DT
        
        delta_v -= (friction + drag + turn_drag)

        self.speed = float(np.clip(self.speed + delta_v, 0.0, MAX_SPEED))

        # Bicycle model heading update
        if abs(steer_angle) > 1e-6:
            self.heading += (self.speed / WHEELBASE) * np.tan(steer_angle) * DT
        self.heading = float(self.heading % (2 * np.pi))

        # Position update
        self.x += self.speed * np.cos(self.heading) * DT
        self.y += self.speed * np.sin(self.heading) * DT

    # ── Step 3b: 7-ray sensor suite ───────────────────────────────────────────
    def get_sensor_readings(self, track) -> np.ndarray:
        """Returns (N_SENSORS,) array, values in [0, 1]."""
        readings = np.array([
            self._cast_ray(a, track) / SENSOR_MAX_DIST
            for a in SENSOR_ANGLES
        ], dtype=np.float32)
        return np.clip(readings, 0.0, 1.0)

    # ── Step 3c: ray-segment intersection ────────────────────────────────────
    def _cast_ray(self, angle_deg: float, track) -> float:
        """
        Cast a ray at (heading + angle_deg) from the car's position.
        Returns distance to nearest boundary hit, or SENSOR_MAX_DIST.
        """
        angle_rad = self.heading + np.deg2rad(angle_deg)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        # Ray: P(t) = (x + t*dx, y + t*dy),  t in [0, SENSOR_MAX_DIST]
        ray_end = np.array([
            self.x + dx * SENSOR_MAX_DIST,
            self.y + dy * SENSOR_MAX_DIST,
        ])
        ray_start = np.array([self.x, self.y])

        min_dist = SENSOR_MAX_DIST

        for boundary in (track.left_bound, track.right_bound):
            for i in range(len(boundary) - 1):
                p = boundary[i]
                q = boundary[i + 1]
                d = self._segment_intersect(ray_start, ray_end, p, q)
                if d is not None and d < min_dist:
                    min_dist = d

        return float(min_dist)

    @staticmethod
    def _segment_intersect(p1, p2, p3, p4):
        """
        Returns distance from p1 along (p2-p1) to intersection with segment p3-p4,
        or None if no intersection within both segments.
        """
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return None  # parallel

        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross

        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return float(t * np.linalg.norm(d1))
        return None

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading, self.speed], dtype=np.float32)