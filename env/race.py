# ── env/race.py ───────────────────────────────────────────────────────────────
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.track  import Track
from env.car    import Car
from env.reward import compute_reward
from config import (TRACK_FILE, MAX_SPEED, N_SENSORS,
                    SENSOR_ANGLES, SENSOR_MAX_DIST,
                    WINDOW_W, WINDOW_H, FPS, HEADLESS)


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.track       = Track(TRACK_FILE)
        self.car         = Car(*self.track.start_pos, self.track.start_heading)
        # Force render_mode off if running headless
        self.render_mode = None if HEADLESS else render_mode

        self.observation_space = spaces.Box(low=-1., high=1., shape=(N_SENSORS+1,), dtype=np.float32)
        self.action_space      = spaces.Box(low=-1., high=1., shape=(2,),           dtype=np.float32)

        self._prev_waypoint       = 0
        self._prev_progress       = 0.0
        self._step_count          = 0
        self._steps_since_last_wp = 0
        self.MAX_STEPS            = 5000
        self._screen = self._clock = self._font = None

        if not HEADLESS:
            self._compute_screen_transform()
        if self.render_mode == "human":
            self._init_pygame()

    # ── transform ─────────────────────────────────────────────────────────────
    def _compute_screen_transform(self):
        all_pts = np.vstack([self.track.centerline, self.track.left_bound, self.track.right_bound])
        pad = 60
        mn, mx = all_pts.min(0), all_pts.max(0)
        span   = mx - mn
        scale  = min((WINDOW_W - 2*pad) / max(span[0], 1e-6),
                     (WINDOW_H - 2*pad) / max(span[1], 1e-6))
        self._scale  = scale
        self._offset = np.array([(WINDOW_W - span[0]*scale)/2 - mn[0]*scale,
                                 (WINDOW_H - span[1]*scale)/2 - mn[1]*scale])
        avg_w_m      = float(np.mean(self.track._w_right + self.track._w_left))
        self._car_px = max(4, int(avg_w_m * scale * 0.4))

    def _to_screen(self, x, y):
        px = int(x * self._scale + self._offset[0])
        py = int(WINDOW_H - (y * self._scale + self._offset[1]))
        return px, py

    # ── gym ───────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car.reset(*self.track.start_pos, self.track.start_heading)
        self._prev_waypoint = self._prev_progress = self._step_count = 0
        self._steps_since_last_wp = 0
        return self._get_obs(), {}

    def step(self, action):
        self.car.step(float(action[0]), float(action[1]))
        self._step_count += 1
        on_track      = self.track.is_on_track(self.car.x, self.car.y)
        curr_wp       = self.track.nearest_waypoint(self.car.x, self.car.y)
        curr_progress = self.track.progress(self.car.x, self.car.y)
        new_wps       = (curr_wp - self._prev_waypoint) % len(self.track.centerline)

        if new_wps > 0:
            self._steps_since_last_wp = 0
        else:
            self._steps_since_last_wp += 1

        reward = compute_reward(on_track, new_wps, self.car.speed,
                                self._prev_progress, curr_progress)
        self._prev_waypoint = curr_wp
        self._prev_progress = curr_progress
        info = {"speed": self.car.speed, "waypoint": curr_wp,
                "progress": curr_progress, "on_track": on_track}

        terminated = not on_track
        if self._steps_since_last_wp > 100:
            terminated = True
            reward -= 5.0
            info["on_track"] = False

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, self._step_count >= self.MAX_STEPS, info

    def _get_obs(self):
        s = self.car.get_sensor_readings(self.track) * 2.0 - 1.0
        v = float(np.clip(self.car.speed / MAX_SPEED, 0, 1)) * 2.0 - 1.0
        return np.append(s, v).astype(np.float32)

    # ── pygame ────────────────────────────────────────────────────────────────
    def _init_pygame(self):
        import pygame
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("F1 Racing Line RL")
            self._screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            self._clock  = pygame.time.Clock()
            self._font   = pygame.font.SysFont("monospace", 15)

    def render(self):
        if HEADLESS:
            return
        import pygame
        if self._screen is None:
            self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._screen.fill((20, 20, 30))

        rb = [self._to_screen(*p) for p in self.track.right_bound]
        lb = [self._to_screen(*p) for p in self.track.left_bound]
        if len(rb) > 2:
            pygame.draw.polygon(self._screen, (55, 55, 65), rb + lb[::-1])
        pygame.draw.lines(self._screen, (210, 210, 210), True, rb, 2)
        pygame.draw.lines(self._screen, (210, 210, 210), True, lb, 2)

        cl = [self._to_screen(*p) for p in self.track.centerline]
        for i in range(0, len(cl)-1, 5):
            pygame.draw.line(self._screen, (70, 70, 110), cl[i], cl[(i+1) % len(cl)], 1)

        cx, cy = self._to_screen(self.car.x, self.car.y)
        for ang in SENSOR_ANGLES:
            ra   = self.car.heading + math.radians(ang)
            dist = self.car._cast_ray(ang, self.track)
            ex, ey = self.car.x + math.cos(ra)*dist, self.car.y + math.sin(ra)*dist
            esx, esy = self._to_screen(ex, ey)
            pygame.draw.line(self._screen, (200, 160, 0), (cx, cy), (esx, esy), 1)
            pygame.draw.circle(self._screen, (255, 220, 0), (esx, esy), 3)

        h, sz = self.car.heading, self._car_px
        pts = [
            (cx + sz   * math.cos(h),        cy - sz   * math.sin(h)),
            (cx + sz*.6* math.cos(h + 2.4),  cy - sz*.6* math.sin(h + 2.4)),
            (cx + sz*.6* math.cos(h - 2.4),  cy - sz*.6* math.sin(h - 2.4)),
        ]
        pygame.draw.polygon(self._screen, (255, 60, 60), [(int(x), int(y)) for x,y in pts])
        pygame.draw.circle(self._screen, (255, 255, 255), (cx, cy), 3)

        on_track = self.track.is_on_track(self.car.x, self.car.y)
        hud = [
            f"Speed:    {self.car.speed * 3.6:.1f} km/h",
            f"Waypoint: {self._prev_waypoint} / {len(self.track.centerline)}",
            f"Step:     {self._step_count}",
            f"On track: {on_track}",
        ]
        for i, line in enumerate(hud):
            col = (100,255,100) if (i==3 and on_track) else \
                  (255,100,100) if (i==3 and not on_track) else (200,200,200)
            self._screen.blit(self._font.render(line, True, col), (10, 10 + i*20))

        pygame.display.flip()
        self._clock.tick(FPS)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None