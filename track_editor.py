import argparse
import csv
import math
import os

import numpy as np
import pygame
from scipy.interpolate import splprep, splev


WINDOW_W = 1280
WINDOW_H = 840
POINT_RADIUS = 7
SELECT_RADIUS_PX = 14
MIN_ZOOM = 1.0
MAX_ZOOM = 40.0
GRID_TARGET_PX = 90
STATUS_FRAMES = 180


class TrackEditor:
    def __init__(self, path: str, default_width: float = 12.0, preview_points: int = 300):
        self.path = path
        self.preview_points = preview_points
        self.points = []
        self.widths = []
        self.default_width = max(1.0, float(default_width))

        self.camera_center = np.zeros(2, dtype=float)
        self.zoom = 4.0
        self.selected_idx = None
        self.dragging_idx = None
        self.panning = False
        self.history = []
        self.show_help = True
        self.status_text = ""
        self.status_frames = 0

        if os.path.exists(self.path):
            self._load_csv(self.path)
            self._set_status(f"Loaded {os.path.basename(self.path)}")
        else:
            self._set_status(f"Creating new track: {os.path.basename(self.path)}")

        pygame.init()
        pygame.display.set_caption("Track Editor")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 16)
        self.font_lg = pygame.font.SysFont("consolas", 20, bold=True)

        self._fit_view()

    def _set_status(self, text: str):
        self.status_text = text
        self.status_frames = STATUS_FRAMES
        print(text)

    def _push_history(self):
        snapshot = (
            [p.copy() for p in self.points],
            [w.copy() for w in self.widths],
            self.selected_idx,
        )
        self.history.append(snapshot)
        if len(self.history) > 100:
            self.history.pop(0)

    def _undo(self):
        if not self.history:
            self._set_status("Nothing to undo")
            return
        points, widths, selected_idx = self.history.pop()
        self.points = [p.copy() for p in points]
        self.widths = [w.copy() for w in widths]
        self.selected_idx = selected_idx
        self.dragging_idx = None
        self._set_status("Undo")

    def _load_csv(self, path: str):
        self.points = []
        self.widths = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.points.append(np.array([float(row["x_m"]), float(row["y_m"])], dtype=float))
                right = float(row["w_tr_right_m"])
                left = float(row["w_tr_left_m"])
                self.widths.append(np.array([right, left], dtype=float))

        if self.widths:
            self.default_width = float(np.mean([w.mean() for w in self.widths]))

    def _save_csv(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_m", "y_m", "w_tr_right_m", "w_tr_left_m"])
            for point, width in zip(self.points, self.widths):
                writer.writerow([
                    f"{point[0]:.3f}",
                    f"{point[1]:.3f}",
                    f"{width[0]:.3f}",
                    f"{width[1]:.3f}",
                ])
        self._set_status(f"Saved {len(self.points)} points to {self.path}")

    def _fit_view(self):
        if not self.points:
            self.camera_center[:] = 0.0
            self.zoom = 4.0
            return

        pts = np.array(self.points)
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = np.maximum(mx - mn, 20.0)
        self.camera_center = (mn + mx) * 0.5
        self.zoom = float(min((WINDOW_W - 120) / span[0], (WINDOW_H - 120) / span[1]))
        self.zoom = float(np.clip(self.zoom, MIN_ZOOM, MAX_ZOOM))

    def _world_to_screen(self, point):
        x = (point[0] - self.camera_center[0]) * self.zoom + WINDOW_W * 0.5
        y = WINDOW_H * 0.5 - (point[1] - self.camera_center[1]) * self.zoom
        return int(round(x)), int(round(y))

    def _screen_to_world(self, pos):
        x = (pos[0] - WINDOW_W * 0.5) / self.zoom + self.camera_center[0]
        y = (WINDOW_H * 0.5 - pos[1]) / self.zoom + self.camera_center[1]
        return np.array([x, y], dtype=float)

    def _zoom_at(self, mouse_pos, wheel_y):
        before = self._screen_to_world(mouse_pos)
        factor = 1.12 ** wheel_y
        self.zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        after = self._screen_to_world(mouse_pos)
        self.camera_center += before - after

    def _nearest_point_idx(self, mouse_pos):
        best_idx = None
        best_dist = SELECT_RADIUS_PX
        for idx, point in enumerate(self.points):
            sx, sy = self._world_to_screen(point)
            dist = math.hypot(mouse_pos[0] - sx, mouse_pos[1] - sy)
            if dist <= best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _insert_after_nearest_segment(self, world_point):
        if len(self.points) < 2:
            return len(self.points)

        pts = self.points
        best_insert = len(pts)
        best_dist = float("inf")
        segment_count = len(pts) if len(pts) >= 3 else len(pts) - 1

        for i in range(segment_count):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            dist = self._point_segment_distance(world_point, a, b)
            if dist < best_dist:
                best_dist = dist
                best_insert = i + 1

        return best_insert

    @staticmethod
    def _point_segment_distance(pt, a, b):
        ab = b - a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-12:
            return float(np.linalg.norm(pt - a))
        t = float(np.clip(np.dot(pt - a, ab) / ab_sq, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(pt - closest))

    def _add_point(self, world_point, insert_mode=False):
        self._push_history()
        insert_at = self._insert_after_nearest_segment(world_point) if insert_mode else len(self.points)
        self.points.insert(insert_at, world_point)
        self.widths.insert(insert_at, np.array([self.default_width, self.default_width], dtype=float))
        self.selected_idx = insert_at
        self._set_status(f"Added point {insert_at}")

    def _delete_selected_or_near(self, mouse_pos):
        idx = self._nearest_point_idx(mouse_pos)
        if idx is None and self.selected_idx is not None and 0 <= self.selected_idx < len(self.points):
            idx = self.selected_idx
        if idx is None:
            self._set_status("No point selected")
            return
        self._push_history()
        self.points.pop(idx)
        self.widths.pop(idx)
        if not self.points:
            self.selected_idx = None
        else:
            self.selected_idx = min(idx, len(self.points) - 1)
        self._set_status(f"Deleted point {idx}")

    def _clear(self):
        if not self.points:
            self._set_status("Track is already empty")
            return
        self._push_history()
        self.points.clear()
        self.widths.clear()
        self.selected_idx = None
        self.dragging_idx = None
        self._set_status("Cleared all points")

    def _adjust_width(self, delta: float):
        delta = float(delta)
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.widths):
            self._push_history()
            self.widths[self.selected_idx] = np.maximum(1.0, self.widths[self.selected_idx] + delta)
            width = self.widths[self.selected_idx]
            self._set_status(
                f"Point {self.selected_idx} width -> R {width[0]:.1f} m, L {width[1]:.1f} m"
            )
            return

        self.default_width = max(1.0, self.default_width + delta)
        self._set_status(f"Default width -> {self.default_width:.1f} m")

    def _build_preview(self):
        if len(self.points) < 4:
            return None

        try:
            xy = np.array(self.points, dtype=float)
            x = np.append(xy[:, 0], xy[0, 0])
            y = np.append(xy[:, 1], xy[0, 1])
            tck, _ = splprep([x, y], s=10.0, per=True, k=3)
            u_new = np.linspace(0, 1, self.preview_points, endpoint=False)
            sx, sy = splev(u_new, tck)
            centerline = np.column_stack([sx, sy])

            width_arr = np.array(self.widths, dtype=float)
            old_u = np.linspace(0, 1, len(width_arr), endpoint=False)
            right = np.interp(u_new, old_u, width_arr[:, 0], period=1.0)
            left = np.interp(u_new, old_u, width_arr[:, 1], period=1.0)

            tangents = np.roll(centerline, -1, axis=0) - centerline
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents = tangents / np.where(norms > 1e-9, norms, 1e-9)
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
            right_bound = centerline + normals * right[:, None]
            left_bound = centerline - normals * left[:, None]
            return centerline, right_bound, left_bound
        except Exception:
            return None

    def _nice_grid_step(self):
        world_step = GRID_TARGET_PX / max(self.zoom, 1e-6)
        magnitude = 10 ** math.floor(math.log10(max(world_step, 1e-6)))
        for factor in (1, 2, 5, 10):
            candidate = factor * magnitude
            if candidate * self.zoom >= GRID_TARGET_PX:
                return candidate
        return 10 * magnitude

    def _draw_grid(self):
        self.screen.fill((12, 14, 22))
        step = self._nice_grid_step()

        left_world = self.camera_center[0] - WINDOW_W * 0.5 / self.zoom
        right_world = self.camera_center[0] + WINDOW_W * 0.5 / self.zoom
        bottom_world = self.camera_center[1] - WINDOW_H * 0.5 / self.zoom
        top_world = self.camera_center[1] + WINDOW_H * 0.5 / self.zoom

        start_x = math.floor(left_world / step) * step
        end_x = math.ceil(right_world / step) * step
        start_y = math.floor(bottom_world / step) * step
        end_y = math.ceil(top_world / step) * step

        grid_col = (30, 34, 50)
        axis_col = (70, 78, 110)
        label_col = (95, 100, 140)

        x = start_x
        while x <= end_x:
            sx, _ = self._world_to_screen((x, 0.0))
            pygame.draw.line(self.screen, axis_col if abs(x) < 1e-9 else grid_col, (sx, 0), (sx, WINDOW_H))
            if abs(x) >= 1e-9:
                text = self.font_sm.render(f"{x:.0f}", True, label_col)
                self.screen.blit(text, (sx + 4, WINDOW_H - 24))
            x += step

        y = start_y
        while y <= end_y:
            _, sy = self._world_to_screen((0.0, y))
            pygame.draw.line(self.screen, axis_col if abs(y) < 1e-9 else grid_col, (0, sy), (WINDOW_W, sy))
            if abs(y) >= 1e-9:
                text = self.font_sm.render(f"{y:.0f}", True, label_col)
                self.screen.blit(text, (8, sy + 2))
            y += step

    def _draw_preview(self):
        preview = self._build_preview()
        if preview is None:
            return

        centerline, right_bound, left_bound = preview
        rb = [self._world_to_screen(p) for p in right_bound]
        lb = [self._world_to_screen(p) for p in left_bound]
        cl = [self._world_to_screen(p) for p in centerline]

        if len(rb) > 2:
            asphalt = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            pygame.draw.polygon(asphalt, (52, 54, 72, 200), rb + lb[::-1])
            self.screen.blit(asphalt, (0, 0))

        if len(rb) > 1:
            pygame.draw.lines(self.screen, (235, 235, 240), True, rb, 3)
            pygame.draw.lines(self.screen, (235, 235, 240), True, lb, 3)

        if len(cl) > 1:
            pygame.draw.lines(self.screen, (210, 190, 80), True, cl, 2)

    def _draw_control_polygon(self):
        if len(self.points) > 1:
            pts = [self._world_to_screen(p) for p in self.points]
            pygame.draw.lines(self.screen, (100, 180, 255), False, pts, 2)
            if len(pts) >= 3:
                pygame.draw.line(self.screen, (70, 120, 170), pts[-1], pts[0], 1)

        for idx, point in enumerate(self.points):
            pos = self._world_to_screen(point)
            colour = (120, 255, 120) if idx == 0 else (240, 240, 240)
            radius = POINT_RADIUS
            if idx == self.selected_idx:
                pygame.draw.circle(self.screen, (255, 215, 90), pos, radius + 5, 2)
                colour = (255, 215, 90)
                radius = radius + 1
            pygame.draw.circle(self.screen, colour, pos, radius)
            label = self.font_sm.render(str(idx), True, (220, 220, 220))
            self.screen.blit(label, (pos[0] + 10, pos[1] - 18))

    def _draw_hud(self):
        panel = pygame.Surface((365, 170), pygame.SRCALPHA)
        panel.fill((8, 10, 18, 215))
        pygame.draw.rect(panel, (70, 76, 110), (0, 0, 365, 170), 1)
        self.screen.blit(panel, (10, 10))

        title = self.font_lg.render("TRACK EDITOR", True, (135, 170, 255))
        self.screen.blit(title, (22, 18))

        selected_width = None
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.widths):
            selected_width = self.widths[self.selected_idx]

        lines = [
            f"File: {self.path}",
            f"Points: {len(self.points)}",
            f"Zoom: {self.zoom:.2f} px/m",
            f"Default width: {self.default_width:.1f} m",
            "Selected: none" if self.selected_idx is None else f"Selected: point {self.selected_idx}",
        ]
        if selected_width is not None:
            lines.append(f"Point width: R {selected_width[0]:.1f}  L {selected_width[1]:.1f}")

        y = 48
        for line in lines:
            surf = self.font_sm.render(line, True, (220, 225, 240))
            self.screen.blit(surf, (22, y))
            y += 22

        if self.status_frames > 0:
            status = self.font_sm.render(self.status_text, True, (255, 225, 140))
            self.screen.blit(status, (22, 142))

        if not self.show_help:
            return

        help_panel = pygame.Surface((440, 242), pygame.SRCALPHA)
        help_panel.fill((8, 10, 18, 215))
        pygame.draw.rect(help_panel, (70, 76, 110), (0, 0, 440, 242), 1)
        self.screen.blit(help_panel, (WINDOW_W - 450, 10))

        help_title = self.font_lg.render("CONTROLS", True, (135, 170, 255))
        self.screen.blit(help_title, (WINDOW_W - 438, 18))

        controls = [
            "Left click empty : add point at end",
            "Shift + Left click : insert after nearest segment",
            "Left drag point : move point",
            "Right click point : delete point",
            "Middle drag : pan camera",
            "Mouse wheel : zoom",
            "S : save CSV",
            "U : undo",
            "F : fit view",
            "[ / ] : shrink or widen selected point",
            "If nothing selected, [ / ] changes default width",
            "Backspace : delete selected point",
            "C : clear all points",
            "H : toggle this help",
            "Esc : quit",
        ]

        y = 48
        for line in controls:
            surf = self.font_sm.render(line, True, (220, 225, 240))
            self.screen.blit(surf, (WINDOW_W - 438, y))
            y += 16

    def draw(self):
        self._draw_grid()
        self._draw_preview()
        self._draw_control_polygon()
        self._draw_hud()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_h:
                self.show_help = not self.show_help
            elif event.key == pygame.K_s:
                self._save_csv()
            elif event.key == pygame.K_u:
                self._undo()
            elif event.key == pygame.K_f:
                self._fit_view()
                self._set_status("Fit view")
            elif event.key == pygame.K_c:
                self._clear()
            elif event.key == pygame.K_BACKSPACE:
                self._delete_selected_or_near((-10_000, -10_000))
            elif event.key == pygame.K_LEFTBRACKET:
                self._adjust_width(-1.0)
            elif event.key == pygame.K_RIGHTBRACKET:
                self._adjust_width(1.0)

        elif event.type == pygame.MOUSEWHEEL:
            self._zoom_at(pygame.mouse.get_pos(), event.y)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 2:
                self.panning = True
            elif event.button == 1:
                idx = self._nearest_point_idx(event.pos)
                if idx is not None:
                    self.selected_idx = idx
                    self.dragging_idx = idx
                    self._push_history()
                else:
                    insert_mode = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    self._add_point(self._screen_to_world(event.pos), insert_mode=insert_mode)
            elif event.button == 3:
                self._delete_selected_or_near(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging_idx = None
            elif event.button == 2:
                self.panning = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_idx is not None and 0 <= self.dragging_idx < len(self.points):
                self.points[self.dragging_idx] = self._screen_to_world(event.pos)
            elif self.panning:
                dx, dy = event.rel
                self.camera_center[0] -= dx / self.zoom
                self.camera_center[1] += dy / self.zoom

        return True

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

            if self.status_frames > 0:
                self.status_frames -= 1

        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Point-and-click track editor")
    parser.add_argument(
        "--file",
        default="data/tracks/custom_track.csv",
        help="CSV file to edit or create",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=12.0,
        help="Default width in metres for new points",
    )
    parser.add_argument(
        "--preview-points",
        type=int,
        default=300,
        help="Number of spline samples used for preview rendering",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    editor = TrackEditor(args.file, default_width=args.width, preview_points=args.preview_points)
    editor.run()


if __name__ == "__main__":
    main()
