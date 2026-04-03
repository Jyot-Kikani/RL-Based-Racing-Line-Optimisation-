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
                    WINDOW_W, WINDOW_H, FPS, HEADLESS, RENDER_CAMERA_MODE)


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
        # Visual-only smoothed state — does NOT affect physics
        self._render_heading  = None   # smoothed heading for sprite
        self._render_ray_ends = None   # smoothed ray endpoints (world-space)
        self._camera_center   = None   # smoothed follow-camera target (world-space)

        if not HEADLESS:
            self._compute_screen_transform()
        if self.render_mode == "human":
            self._init_pygame()

    # ── transform ─────────────────────────────────────────────────────────────
    def _compute_screen_transform(self):
        if RENDER_CAMERA_MODE == "full":
            self._compute_full_track_transform()
            return

        # Fixed local zoom window for the follow camera.
        # SENSOR_MAX_DIST already defines how far the agent can "see", so we
        # render a little beyond that to keep the upcoming road readable.
        pad      = 36
        view_w_m = SENSOR_MAX_DIST * 2.4
        view_h_m = view_w_m * (WINDOW_H / WINDOW_W)
        scale    = min((WINDOW_W - 2 * pad) / max(view_w_m, 1e-6),
                       (WINDOW_H - 2 * pad) / max(view_h_m, 1e-6))
        self._scale  = scale
        self._offset = np.zeros(2, dtype=float)
        avg_w_m      = float(np.mean(self.track._w_right + self.track._w_left))
        self._car_px = max(4, int(avg_w_m * scale * 0.4))

    def _compute_full_track_transform(self):
        all_pts = np.vstack([self.track.centerline, self.track.left_bound, self.track.right_bound])
        pad = 60
        mn, mx = all_pts.min(0), all_pts.max(0)
        span   = mx - mn
        scale  = min((WINDOW_W - 2 * pad) / max(span[0], 1e-6),
                     (WINDOW_H - 2 * pad) / max(span[1], 1e-6))
        self._scale  = scale
        self._offset = np.array([(WINDOW_W - span[0] * scale) / 2 - mn[0] * scale,
                                 (WINDOW_H - span[1] * scale) / 2 - mn[1] * scale])
        avg_w_m      = float(np.mean(self.track._w_right + self.track._w_left))
        self._car_px = max(4, int(avg_w_m * scale * 0.4))

    def _update_camera_transform(self):
        if RENDER_CAMERA_MODE == "full":
            return

        heading = self._render_heading if self._render_heading is not None else self.car.heading
        lookahead_m = 22.0 + 0.25 * self.car.speed
        target = np.array([
            self.car.x + math.cos(heading) * lookahead_m,
            self.car.y + math.sin(heading) * lookahead_m,
        ], dtype=float)

        if self._camera_center is None:
            self._camera_center = target
        else:
            self._camera_center += (target - self._camera_center) * 0.12

        self._offset = np.array([
            WINDOW_W * 0.5 - self._camera_center[0] * self._scale,
            WINDOW_H * 0.5 - self._camera_center[1] * self._scale,
        ], dtype=float)

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
        self._render_heading = None
        self._render_ray_ends = None
        self._camera_center = None
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
    # Supersampling factor: render internally at SSAA× resolution, then
    # smoothscale down to the window. Eliminates pixel-jitter on sub-pixel motion.
    # 2 = good quality,  can raise to 3 on fast machines.  1 = off.
    _SSAA = 1

    # Visual-only heading smoother (EMA).
    # 0.0 = frozen, 1.0 = no smoothing.  Lower = smoother but more lag.
    _HEADING_SMOOTH = 0.18

    def _init_pygame(self):
        import pygame
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("F1 Racing Line  \u00b7  RL Agent")
            self._screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            self._clock   = pygame.time.Clock()
            # Fonts stay at display resolution (HUD is blitted post-downscale)
            self._font_sm = pygame.font.SysFont("monospace", 13)
            self._font_lg = pygame.font.SysFont("monospace", 17, bold=True)
            self._font    = self._font_sm
            # Internal hi-res canvas: all track/car drawing goes here
            cw = WINDOW_W * self._SSAA
            ch = WINDOW_H * self._SSAA
            self._canvas   = pygame.Surface((cw, ch))
            self._ray_surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
            # Car sprite — loaded once, rotated each frame via rotozoom
            import os
            sprite_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "assets", "mercedes.png")
            )
            self._car_sprite = pygame.image.load(sprite_path).convert_alpha()

    # ── drawing helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _lerp_colour(c1, c2, t):
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _to_canvas(self, x, y):
        """World coords → canvas pixel coords (SSAA× resolution)."""
        sx = x * self._scale + self._offset[0]
        sy = WINDOW_H - (y * self._scale + self._offset[1])
        return int(sx * self._SSAA), int(sy * self._SSAA)

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

        S = self._SSAA                   # scale multiplier shorthand
        cv = self._canvas                # hi-res draw target

        on_track  = self.track.is_on_track(self.car.x, self.car.y)
        speed_kmh = self.car.speed * 3.6
        spd_t     = min(1.0, self.car.speed / MAX_SPEED)

        # Initialise smoothed heading on the very first frame (before rays or car use it)
        if self._render_heading is None:
            self._render_heading = self.car.heading
        self._update_camera_transform()

        # ── 1. Background ─────────────────────────────────────────────────────
        cv.fill((10, 12, 20))

        # ── 2. Track boundary lists (canvas coords) ───────────────────────────
        rb = [self._to_canvas(*p) for p in self.track.right_bound]
        lb = [self._to_canvas(*p) for p in self.track.left_bound]

        # ── 3. Asphalt fill ───────────────────────────────────────────────────
        if len(rb) > 2:
            pygame.draw.polygon(cv, (38, 38, 52), rb + lb[::-1])

        # ── 4. Centerline list ────────────────────────────────────────────────
        cl = [self._to_canvas(*p) for p in self.track.centerline]

        # ── 5. Kerb lines — alternating white / red ───────────────────────────
        kerb_white = (230, 230, 230)
        kerb_red   = (210,  40,  40)
        for boundary in (rb, lb):
            for i in range(len(boundary) - 1):
                colour = kerb_white if (i // 3) % 2 == 0 else kerb_red
                pygame.draw.line(cv, colour, boundary[i], boundary[i + 1], S * 2)

        # ── 6. Centerline dashes ──────────────────────────────────────────────
        for i in range(0, len(cl) - 1, 1):
            if (i // 4) % 2 == 0:
                pygame.draw.line(cv, (200, 190, 80), cl[i], cl[i + 1], S)

        # ── 7. Sensor rays — EMA-smoothed endpoints ───────────────────────────
        # Step 1: compute actual endpoints using smoothed heading (not raw)
        cx, cy = self._to_canvas(self.car.x, self.car.y)
        actual_ends = []
        for ang in SENSOR_ANGLES:
            ra   = self._render_heading + math.radians(ang)  # smoothed heading
            dist = self.car._cast_ray(ang, self.track)
            actual_ends.append((self.car.x + math.cos(ra) * dist,
                                 self.car.y + math.sin(ra) * dist))

        # Step 2: EMA-blend endpoints in world space (same alpha as heading)
        if self._render_ray_ends is None:
            self._render_ray_ends = actual_ends[:]
        else:
            a = self._HEADING_SMOOTH
            self._render_ray_ends = [
                (rx + (ax - rx) * a, ry + (ay - ry) * a)
                for (rx, ry), (ax, ay) in zip(self._render_ray_ends, actual_ends)
            ]

        # Step 3: draw with smoothed endpoints; derive t from smoothed distance
        self._ray_surf.fill((0, 0, 0, 0))
        for ex, ey in self._render_ray_ends:
            esx, esy    = self._to_canvas(ex, ey)
            smooth_dist = math.hypot(ex - self.car.x, ey - self.car.y)
            t           = max(0.0, 1.0 - smooth_dist / SENSOR_MAX_DIST)
            ray_col     = self._lerp_colour((0, 200, 255), (255, 60, 60), t)
            pygame.draw.line(self._ray_surf, (*ray_col, 90), (cx, cy), (esx, esy), S)
            dot_r = max(S, int((3 + t * 3) * S))
            pygame.draw.circle(self._ray_surf, (*ray_col, 160), (esx, esy), dot_r)
        cv.blit(self._ray_surf, (0, 0))


        # ── 8. Car sprite — EMA-smoothed heading, then rotozoom ──────────────
        actual_h = self.car.heading   # raw physics heading (radians)

        # Shortest angular path (handles 0/2π wrap-around correctly)
        diff = (actual_h - self._render_heading + math.pi) % (2 * math.pi) - math.pi
        self._render_heading = (self._render_heading + diff * self._HEADING_SMOOTH) % (2 * math.pi)

        # pygame.transform.rotozoom uses CCW degrees
        heading_deg = math.degrees(self._render_heading)

        target_w_px = self._car_px * S * 2
        scale_f     = target_w_px / max(self._car_sprite.get_width(), 1)

        rotated = pygame.transform.rotozoom(self._car_sprite, heading_deg, scale_f)
        cx, cy  = self._to_canvas(self.car.x, self.car.y)
        cv.blit(rotated, (cx - rotated.get_width() // 2, cy - rotated.get_height() // 2))

        # ── 9. Downscale canvas → window (this is where the AA happens) ───────
        pygame.transform.smoothscale(cv, (WINDOW_W, WINDOW_H), self._screen)

        # ── 10. HUD — drawn directly on screen after downscale ────────────────
        # (Fonts are at display resolution; drawing pre-scale would make them blurry)
        progress = self._prev_waypoint / max(1, len(self.track.centerline))

        panel_surf = pygame.Surface((230, 108), pygame.SRCALPHA)
        panel_surf.fill((8, 10, 22, 200))
        pygame.draw.rect(panel_surf, (60, 60, 120, 180), (0, 0, 230, 108), 1)
        self._screen.blit(panel_surf, (8, 8))

        title_surf = self._font_lg.render("RL AGENT", True, (140, 140, 255))
        self._screen.blit(title_surf, (16, 13))

        spd_col  = self._lerp_colour((100, 220, 100), (255, 80, 80), spd_t)
        self._screen.blit(self._font_lg.render(f"{speed_kmh:5.1f} km/h", True, spd_col), (16, 33))

        bar_x, bar_y, bar_w, bar_h = 16, 54, 210, 8
        pygame.draw.rect(self._screen, (30, 30, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * spd_t)
        if fill_w > 0:
            pygame.draw.rect(self._screen, spd_col, (bar_x, bar_y, fill_w, bar_h), border_radius=4)

        self._screen.blit(
            self._font_sm.render(f"Lap  {self._prev_waypoint:4d}/{len(self.track.centerline)}", True, (180, 180, 220)),
            (16, 67))
        pygame.draw.rect(self._screen, (30, 30, 50), (bar_x, 82, bar_w, 6), border_radius=3)
        prog_w = int(bar_w * progress)
        if prog_w > 0:
            pygame.draw.rect(self._screen, (100, 160, 255), (bar_x, 82, prog_w, 6), border_radius=3)

        step_col  = (160, 160, 200)
        track_col = (80, 255, 120) if on_track else (255, 80, 80)
        self._screen.blit(self._font_sm.render(f"Step {self._step_count:5d}", True, step_col),   (16,  92))
        self._screen.blit(self._font_sm.render("ON TRACK" if on_track else "OFF TRACK", True, track_col), (130, 92))

        pygame.display.flip()
        self._clock.tick(FPS)

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
