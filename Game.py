"""
Game.py
=======
Scalable 2D maze navigation game with reinforcement learning (REINFORCE),
an independent 2D camera, minimap, path persistence, and a training dashboard.

Controls:
    F              toggle follow-agent camera
    WASD           pan camera (free mode)
    Mouse drag     pan camera
    Mouse wheel    zoom (viewport) / scroll (dashboard)
    SPACE          centre camera on agent
    M              toggle minimap
    P              pause / resume training
    T              start / stop training
    E              toggle evaluation mode
    G              generate new random map
    N              reset current episode
    K              save model
    L              load model
    V              toggle path overlay
    1/2/3          simulation speed (1x, 4x, 16x)
    X              toggle user-control of the agent
    Arrow keys     control the agent when user-control is ON
    ESC            exit edit mode (or quit)

When user-control is ON the agent renders in sky blue; when OFF it renders
in its original accent colour.  Rewards, trajectory recording, and policy
updates all use the exact same pipeline as before — actions chosen by the
user are still pushed through MazeEnv.step() and, in training mode, are
learned from just like actions sampled from the policy.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
import tensorflow as tf

import model as rl_model

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #
STATE_DIM = rl_model.STATE_DIM
ACTION_DIM = rl_model.ACTION_DIM

EPISODES = 500
LEARNING_RATE = 0.001
GAMMA = 0.99
ENTROPY_COEF = 0.01
MAX_STEPS = 300

WINDOW_W, WINDOW_H = 1280, 600
DASHBOARD_W = 340
FPS = 60

CELL = 24
MIN_ZOOM, MAX_ZOOM = 0.15, 4.0
PAN_SPEED = 600.0

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MAPS_DIR = BASE_DIR / "maps"
PATHS_DIR = BASE_DIR / "paths"
LOGS_DIR = BASE_DIR / "logs"
for _d in (MODELS_DIR, MAPS_DIR, PATHS_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)

DIR_VECTORS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

ACTION_TABLE: Dict[int, Tuple[int, Tuple[int, int]]] = {
    0: (0, DIR_VECTORS[0]),
    1: (1, DIR_VECTORS[1]),
    2: (2, DIR_VECTORS[2]),
    3: (3, DIR_VECTORS[3]),
}

REWARD_GOAL = 50.0
REWARD_CLOSER = 1.0
REWARD_FARTHER = -1.0
REWARD_WALL = -3.0
REWARD_REVISIT = -2.0
REWARD_STEP = -0.1
REWARD_TIMEOUT = -10.0


# --------------------------------------------------------------------------- #
# COLORS
# --------------------------------------------------------------------------- #
COLORS = {
    "bg":            (18, 20, 26),
    "cell":          (35, 38, 48),
    "wall":          (70, 75, 95),
    "grid":          (28, 30, 38),
    "agent":         (80, 200, 255),      # original accent colour
    "agent_user":    (135, 220, 255),     # sky blue when user is in control
    "start":         (90, 220, 120),
    "goal":          (255, 180, 60),
    "visited":       (50, 60, 85),
    "current_path":  (120, 200, 120),
    "best_path":     (255, 120, 220),

    "dash_bg":       (24, 26, 34),
    "dash_line":     (45, 48, 60),
    "text":          (220, 225, 235),
    "text_dim":      (150, 155, 170),
    "accent":        (80, 200, 255),
    "good":          (100, 220, 130),
    "bad":           (240, 100, 100),

    "button":        (55, 60, 80),
    "button_hover":  (75, 82, 110),
    "input_bg":      (40, 43, 55),
    "input_text":    (240, 240, 245),
    "input_border":  (70, 75, 95),
    "scrollbar":     (80, 90, 120),
    "scrollbar_bg":  (35, 38, 48),
}


# --------------------------------------------------------------------------- #
# Map generation
# --------------------------------------------------------------------------- #
class MazeGenerator:
    EMPTY = 0
    WALL = 1

    _MAX_DFS_NODES = 200_000
    _MAX_DFS_DEPTH = 600
    _MAX_ROUTE_ATTEMPTS = 20

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self.seed)

    def generate(self, width, height, wall_density=0.25,
                 start=None, goal=None, min_routes=1,
                 algorithm="random_walk") -> Dict:
        if width < 3 or height < 3:
            raise ValueError("Map dimensions must be at least 3x3")
        if not (0.0 <= wall_density < 1.0):
            raise ValueError("wall_density must be in [0, 1)")

        grid = np.full((height, width), self.WALL, dtype=np.uint8)

        if start is None: start = (1, 1)
        if goal is None:  goal = (height - 2, width - 2)

        start = (int(np.clip(start[0], 0, height - 1)),
                 int(np.clip(start[1], 0, width - 1)))
        goal = (int(np.clip(goal[0], 0, height - 1)),
                int(np.clip(goal[1], 0, width - 1)))
        if start == goal:
            goal = (min(height - 1, start[0] + 1),
                    min(width - 1, start[1] + 1))
            if goal == start:
                goal = (0, 0) if start != (0, 0) else (height - 1, width - 1)

        self._carve_random_route(grid, start, goal)

        open_prob = max(0.0, 1.0 - wall_density - 0.15)
        extra = self.rng.random((height, width)) < open_prob
        grid[extra] = self.EMPTY

        self._punch_branches(grid, start, goal, wall_density)

        grid[start] = self.EMPTY
        grid[goal] = self.EMPTY

        if not self._is_connected(grid, start, goal):
            self._carve_random_route(grid, start, goal, force=True)

        route_count = 1
        if min_routes > 1:
            route_count = self._count_routes(grid, start, goal,
                                             cap=min_routes + 1)
            attempts = 0
            while route_count < min_routes and attempts < self._MAX_ROUTE_ATTEMPTS:
                self._open_random_connector(grid, start, goal)
                route_count = self._count_routes(grid, start, goal,
                                                 cap=min_routes + 1)
                attempts += 1

        shortest = self._bfs_distance(grid, start, goal)
        if shortest is None:
            raise RuntimeError("Map generation failed to produce a solvable maze")

        return {
            "grid": grid,
            "start": start,
            "goal": goal,
            "seed": int(self.seed),
            "params": {
                "width": int(width),
                "height": int(height),
                "wall_density": float(wall_density),
                "min_routes": int(min_routes),
                "algorithm": algorithm,
            },
            "shortest_distance": int(shortest),
            "route_count": int(route_count),
        }

    def _carve_random_route(self, grid, start, goal, force=False):
        h, w = grid.shape
        r, c = start
        gr, gc = goal
        grid[r, c] = self.EMPTY
        safety = 0
        while (r, c) != (gr, gc) and safety < (h + w) * 4:
            safety += 1
            moves = []
            if gr > r: moves.append((1, 0))
            elif gr < r: moves.append((-1, 0))
            if gc > c: moves.append((0, 1))
            elif gc < c: moves.append((0, -1))
            if not moves: break
            if len(moves) > 1 and self.rng.random() < 0.35:
                dr, dc = moves[self.rng.integers(0, len(moves))]
            else:
                dr, dc = moves[0] if self.rng.random() < 0.5 else moves[-1]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                r, c = nr, nc
                grid[r, c] = self.EMPTY
        if force and grid[gr, gc] != self.EMPTY:
            grid[gr, gc] = self.EMPTY

    def _punch_branches(self, grid, start, goal, wall_density):
        h, w = grid.shape
        n = max(1, int((h * w) * 0.02 * (1.0 - wall_density)))
        for _ in range(n):
            r = int(self.rng.integers(0, h))
            c = int(self.rng.integers(0, w))
            if (r, c) in (start, goal): continue
            grid[r, c] = self.EMPTY

    def _open_random_connector(self, grid, start, goal):
        h, w = grid.shape
        candidates = []
        rows = self.rng.integers(1, max(2, h - 1), size=min(400, h * w))
        cols = self.rng.integers(1, max(2, w - 1), size=min(400, h * w))
        for r, c in zip(rows, cols):
            r, c = int(r), int(c)
            if grid[r, c] != self.WALL: continue
            if (r, c) in (start, goal): continue
            open_nb = 0
            for dr, dc in DIR_VECTORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == self.EMPTY:
                    open_nb += 1
            if open_nb >= 2:
                candidates.append((r, c))
                if len(candidates) >= 16: break
        if candidates:
            r, c = candidates[int(self.rng.integers(0, len(candidates)))]
            grid[r, c] = self.EMPTY

    def _is_connected(self, grid, start, goal):
        return self._bfs_distance(grid, start, goal) is not None

    def _bfs_distance(self, grid, start, goal):
        h, w = grid.shape
        if grid[start] == self.WALL or grid[goal] == self.WALL: return None
        visited = np.zeros((h, w), dtype=bool)
        dq = collections.deque([(start[0], start[1], 0)])
        visited[start] = True
        while dq:
            r, c, d = dq.popleft()
            if (r, c) == goal: return d
            for dr, dc in DIR_VECTORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w \
                        and not visited[nr, nc] and grid[nr, nc] == self.EMPTY:
                    visited[nr, nc] = True
                    dq.append((nr, nc, d + 1))
        return None

    def _count_routes(self, grid, start, goal, cap=4):
        h, w = grid.shape
        if grid[start] == self.WALL or grid[goal] == self.WALL: return 0
        visited = np.zeros((h, w), dtype=bool)
        visited[start] = True
        count = 0
        nodes = 0
        hit_budget = False

        def dfs(r, c, depth):
            nonlocal count, nodes, hit_budget
            if hit_budget or count >= cap: return
            nodes += 1
            if nodes >= self._MAX_DFS_NODES or depth > self._MAX_DFS_DEPTH:
                hit_budget = True; return
            if (r, c) == goal:
                count += 1; return
            for dr, dc in DIR_VECTORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w \
                        and not visited[nr, nc] and grid[nr, nc] == self.EMPTY:
                    visited[nr, nc] = True
                    dfs(nr, nc, depth + 1)
                    visited[nr, nc] = False
                    if hit_budget or count >= cap: return

        dfs(start[0], start[1], 0)
        if hit_budget and count == 0: return 1
        return count

    def bfs_full_distances(self, grid, goal):
        h, w = grid.shape
        dist = np.full((h, w), -1, dtype=np.int32)
        if grid[goal] == self.WALL: return dist
        dq = collections.deque([goal])
        dist[goal] = 0
        while dq:
            r, c = dq.popleft()
            for dr, dc in DIR_VECTORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w \
                        and dist[nr, nc] == -1 and grid[nr, nc] == self.EMPTY:
                    dist[nr, nc] = dist[r, c] + 1
                    dq.append((nr, nc))
        return dist


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeRecord:
    episode: int
    steps: int
    total_reward: float
    success: bool
    path: List[Tuple[int, int]]
    actions: List[int]
    rewards: List[float]
    policy_loss: float = 0.0
    entropy: float = 0.0


class MazeEnv:
    def __init__(self, config: Dict) -> None:
        self.load_config(config)
        self.reset()

    def load_config(self, config: Dict) -> None:
        self.grid: np.ndarray = np.array(config["grid"], dtype=np.uint8)
        self.start: Tuple[int, int] = tuple(config["start"])
        self.goal: Tuple[int, int] = tuple(config["goal"])
        self.seed: int = int(config.get("seed", 0))
        self.params: Dict = dict(config.get("params", {}))
        self.shortest_distance: int = int(config.get("shortest_distance", -1))
        self.height, self.width = self.grid.shape
        self.dist_to_goal = MazeGenerator().bfs_full_distances(self.grid, self.goal)

    def refresh_distances(self) -> None:
        self.dist_to_goal = MazeGenerator().bfs_full_distances(self.grid, self.goal)
        d = MazeGenerator()._bfs_distance(self.grid, self.start, self.goal)
        self.shortest_distance = int(d) if d is not None else -1

    def reset(self) -> np.ndarray:
        self.agent_r, self.agent_c = self.start
        self.orientation = 0
        self.prev_action = 0
        self.steps = 0
        self.total_reward = 0.0
        self.visited = {(self.agent_r, self.agent_c)}
        self.current_path = [(self.agent_r, self.agent_c)]
        self.done = False
        self.success = False
        self.collisions = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        h, w = self.height, self.width
        r, c = self.agent_r, self.agent_c
        norm_x = c / max(1, w - 1)
        norm_y = r / max(1, h - 1)
        rel_x = (self.goal[1] - c) / max(1, w - 1)
        rel_y = (self.goal[0] - r) / max(1, h - 1)
        orient = [0.0, 0.0, 0.0, 0.0]
        orient[self.orientation] = 1.0
        obstacles = []
        for dr, dc in DIR_VECTORS:
            nr, nc = r + dr, c + dc
            blocked = not (0 <= nr < h and 0 <= nc < w) or self.grid[nr, nc] == 1
            obstacles.append(1.0 if blocked else 0.0)
        if self.shortest_distance > 0:
            norm_dist = self.dist_to_goal[r, c] / self.shortest_distance \
                if self.dist_to_goal[r, c] >= 0 else 1.0
        else:
            norm_dist = 0.0
        prev_action_norm = self.prev_action / 3.0
        return np.array(
            [norm_x, norm_y, rel_x, rel_y, *orient, *obstacles,
             norm_dist, prev_action_norm],
            dtype=np.float32,
        )

    def _orientation_offset(self, action):
        if action == 0: return self.orientation, DIR_VECTORS[self.orientation]
        if action == 1:
            o = (self.orientation + 1) % 4; return o, DIR_VECTORS[o]
        if action == 2:
            o = (self.orientation + 2) % 4; return o, DIR_VECTORS[o]
        if action == 3:
            o = (self.orientation + 3) % 4; return o, DIR_VECTORS[o]
        raise ValueError(f"Invalid action {action}")

    def step(self, action):
        if self.done:
            return self.observe(), 0.0, True, {"info": "episode already done"}
        self.steps += 1
        new_orient, (dr, dc) = self._orientation_offset(action)
        nr, nc = self.agent_r + dr, self.agent_c + dc
        h, w = self.height, self.width
        reward = REWARD_STEP
        info = {"hit_wall": False, "revisit": False, "closer": False}
        blocked = not (0 <= nr < h and 0 <= nc < w) or self.grid[nr, nc] == 1
        if blocked:
            reward += REWARD_WALL
            self.collisions += 1
            info["hit_wall"] = True
            self.orientation = new_orient
        else:
            old_dist = self.dist_to_goal[self.agent_r, self.agent_c]
            self.agent_r, self.agent_c = nr, nc
            self.orientation = new_orient
            new_dist = self.dist_to_goal[nr, nc]
            if (nr, nc) in self.visited:
                reward += REWARD_REVISIT
                info["revisit"] = True
            else:
                self.visited.add((nr, nc))
                if old_dist >= 0 and new_dist >= 0:
                    if new_dist < old_dist:
                        reward += REWARD_CLOSER
                        info["closer"] = True
                    elif new_dist > old_dist:
                        reward += REWARD_FARTHER
            self.current_path.append((nr, nc))
        self.prev_action = action
        self.total_reward += reward
        if (self.agent_r, self.agent_c) == self.goal:
            reward += REWARD_GOAL
            self.total_reward += REWARD_GOAL
            self.done = True
            self.success = True
        elif self.steps >= MAX_STEPS:
            reward += REWARD_TIMEOUT
            self.total_reward += REWARD_TIMEOUT
            self.done = True
            self.success = False
        return self.observe(), float(reward), self.done, info


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
@dataclass
class Camera:
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0
    follow: bool = True
    view_w: int = WINDOW_W - DASHBOARD_W
    view_h: int = WINDOW_H

    def world_to_screen(self, wx, wy):
        return ((wx - self.x) * self.zoom + self.view_w / 2,
                (wy - self.y) * self.zoom + self.view_h / 2)

    def screen_to_world(self, sx, sy):
        return ((sx - self.view_w / 2) / self.zoom + self.x,
                (sy - self.view_h / 2) / self.zoom + self.y)

    def visible_world_rect(self, cell):
        half_w = (self.view_w / 2) / (self.zoom * cell)
        half_h = (self.view_h / 2) / (self.zoom * cell)
        cx = self.x / cell; cy = self.y / cell
        return (int(math.floor(cx - half_w)) - 1,
                int(math.floor(cy - half_h)) - 1,
                int(math.ceil(cx + half_w)) + 1,
                int(math.ceil(cy + half_h)) + 1)

    def clamp(self, map_w, map_h, cell):
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            self.x = map_w * cell / 2
            self.y = map_h * cell / 2
        max_x = map_w * cell
        max_y = map_h * cell
        self.x = max(0.0, min(max_x, self.x))
        self.y = max(0.0, min(max_y, self.y))
        if not math.isfinite(self.zoom) or self.zoom <= 0:
            self.zoom = 1.0
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))

    def center_on(self, wx, wy):
        self.x, self.y = wx, wy

    def smooth_follow(self, tx, ty, alpha=0.18):
        self.x += (tx - self.x) * alpha
        self.y += (ty - self.y) * alpha

    def pan(self, dx, dy):
        self.x += dx / self.zoom
        self.y += dy / self.zoom

    def zoom_at(self, factor, sx, sy):
        wx, wy = self.screen_to_world(sx, sy)
        new_zoom = self.zoom * factor
        if not math.isfinite(new_zoom) or new_zoom <= 0:
            new_zoom = 1.0
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, new_zoom))
        self.x = wx - (sx - self.view_w / 2) / self.zoom
        self.y = wy - (sy - self.view_h / 2) / self.zoom


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def save_map(config, name):
    path = MAPS_DIR / f"{name}.json"
    data = {
        "id": name,
        "grid": config["grid"].tolist() if isinstance(config["grid"], np.ndarray)
                else config["grid"],
        "start": list(config["start"]),
        "goal": list(config["goal"]),
        "seed": int(config.get("seed", 0)),
        "params": config.get("params", {}),
        "shortest_distance": int(config.get("shortest_distance", -1)),
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_map(name):
    path = MAPS_DIR / f"{name}.json"
    data = json.loads(path.read_text())
    grid = np.array(data["grid"], dtype=np.uint8)
    return {
        "grid": grid,
        "start": tuple(data["start"]),
        "goal": tuple(data["goal"]),
        "seed": int(data.get("seed", 0)),
        "params": data.get("params", {}),
        "shortest_distance": int(data.get("shortest_distance", -1)),
        "id": data.get("id", name),
    }


def list_saved_maps():
    return sorted(p.stem for p in MAPS_DIR.glob("*.json"))


def save_weights(model, optimizer, filename="maze_policy.weights.h5"):
    wpath = MODELS_DIR / filename
    model.save_weights(str(wpath))
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt.write(str(MODELS_DIR / "optimizer.ckpt"))
    return wpath


def load_weights(model, optimizer, filename="maze_policy.weights.h5"):
    wpath = MODELS_DIR / filename
    if not wpath.exists(): return False
    try:
        model.load_weights(str(wpath))
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        latest = tf.train.latest_checkpoint(str(MODELS_DIR))
        if latest:
            try: ckpt.read(latest).expect_partial()
            except Exception: pass
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Path manager
# --------------------------------------------------------------------------- #
class PathManager:
    def __init__(self) -> None:
        self.best_path: Optional[List[Tuple[int, int]]] = None
        self.best_length: int = 10**9
        self.best_episode: int = -1
        self.episodes: List[EpisodeRecord] = []
        self.map_id: str = "unknown"
        self.history_file = PATHS_DIR / "path_history.json"

    def reset_for_map(self, map_id: str) -> None:
        self.map_id = map_id
        self.best_path = None
        self.best_length = 10**9
        self.best_episode = -1
        self.episodes = []
        f = PATHS_DIR / f"best_path_{map_id}.json"
        if f.exists():
            try:
                data = json.loads(f.read_text())
                self.best_path = [tuple(p) for p in data.get("path", [])]
                self.best_length = int(data.get("length", 10**9))
                self.best_episode = int(data.get("episode", -1))
            except Exception:
                pass

    def record_episode(self, record: EpisodeRecord) -> None:
        self.episodes.append(record)
        if record.success:
            length = len(record.path) - 1
            if length < self.best_length:
                self.best_length = length
                self.best_path = list(record.path)
                self.best_episode = record.episode
                self._save_best(record.episode)

    def _save_best(self, episode: int) -> None:
        data = {
            "map_id": self.map_id,
            "episode": episode,
            "length": self.best_length,
            "path": [list(p) for p in self.best_path or []],
        }
        try:
            (PATHS_DIR / f"best_path_{self.map_id}.json")\
                .write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def save_history(self) -> None:
        out = []
        for ep in self.episodes[-50:]:
            out.append({
                "episode": ep.episode,
                "steps": ep.steps,
                "total_reward": ep.total_reward,
                "success": ep.success,
                "path": [list(p) for p in ep.path],
                "actions": ep.actions,
                "rewards": ep.rewards,
            })
        try:
            self.history_file.write_text(json.dumps(out, indent=2))
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
class TextInput:
    def __init__(self, initial=""):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.text = str(initial)
        self.active = False
        self.blink = 0.0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was = self.active
            self.active = self.rect.collidepoint(event.pos)
            if self.active:
                self.blink = 0.0
            return self.active and not was
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_TAB:
                self.active = False
            elif event.unicode.isprintable() and len(self.text) < 8:
                self.text += event.unicode
            return True
        return False

    def draw(self, surf, font, local_rect):
        pygame.draw.rect(surf, COLORS["input_bg"], local_rect, border_radius=4)
        pygame.draw.rect(surf, COLORS["input_border"], local_rect, 1, border_radius=4)
        s = font.render(self.text, True, COLORS["input_text"])
        surf.blit(s, (local_rect.x + 6, local_rect.y + 3))
        if self.active:
            self.blink += 1
            if int(self.blink / 30) % 2 == 0:
                cx = local_rect.x + 6 + font.size(self.text)[0]
                pygame.draw.line(surf, COLORS["input_text"],
                                 (cx, local_rect.y + 4),
                                 (cx, local_rect.bottom - 4), 1)


class Dropdown:
    def __init__(self, items, font):
        self.items = list(items)
        self.font = font
        self.open = False
        self.selected_index = 0 if self.items else -1
        self.box_rect = pygame.Rect(0, 0, 0, 0)
        self.item_rects = []

    @property
    def selected(self):
        if 0 <= self.selected_index < len(self.items):
            return self.items[self.selected_index]
        return None

    def refresh(self, items):
        self.items = list(items)
        self.selected_index = 0 if self.items else -1

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.box_rect.collidepoint(event.pos):
                self.open = not self.open
                return True
            if self.open:
                for i, r in enumerate(self.item_rects):
                    if r.collidepoint(event.pos):
                        self.selected_index = i
                        self.open = False
                        return True
                self.open = False
        return False

    def draw_box(self, surf, local_rect):
        pygame.draw.rect(surf, COLORS["input_bg"], local_rect, border_radius=4)
        pygame.draw.rect(surf, COLORS["input_border"], local_rect, 1, border_radius=4)
        label = self.selected if self.selected else "(empty)"
        s = self.font.render(label, True, COLORS["input_text"])
        surf.blit(s, (local_rect.x + 6, local_rect.y + 3))
        ax = local_rect.right - 14
        ay = local_rect.centery
        pygame.draw.polygon(surf, COLORS["text"],
                            [(ax - 5, ay - 3), (ax + 5, ay - 3), (ax, ay + 3)])

    def draw_list(self, screen, max_visible=8):
        if not self.open or not self.items:
            self.item_rects = []
            return
        n = min(max_visible, len(self.items))
        row_h = self.box_rect.h
        list_rect = pygame.Rect(
            self.box_rect.x, self.box_rect.bottom + 2,
            self.box_rect.w, row_h * n,
        )
        pygame.draw.rect(screen, COLORS["input_bg"], list_rect, border_radius=4)
        pygame.draw.rect(screen, COLORS["input_border"], list_rect, 1, border_radius=4)

        start = max(0, self.selected_index - n // 2)
        self.item_rects = []
        for i in range(n):
            idx = start + i
            if idx >= len(self.items):
                break
            r = pygame.Rect(list_rect.x, list_rect.y + i * row_h,
                            list_rect.w, row_h)
            hover = r.collidepoint(pygame.mouse.get_pos())
            if idx == self.selected_index:
                pygame.draw.rect(screen, COLORS["button_hover"], r)
            elif hover:
                pygame.draw.rect(screen, COLORS["button"], r)
            s = self.font.render(self.items[idx], True, COLORS["input_text"])
            screen.blit(s, (r.x + 6, r.y + 3))
            self.item_rects.append(r)


# --------------------------------------------------------------------------- #
# Game
# --------------------------------------------------------------------------- #
class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Scalable RL Maze Navigation")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_s = pygame.font.SysFont("consolas", 13)
        self.font_m = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_l = pygame.font.SysFont("consolas", 20, bold=True)

        # RL components
        self.model = rl_model.create_policy_model(
            state_dim=STATE_DIM, learning_rate=LEARNING_RATE
        )
        self.optimizer = rl_model.create_optimizer(LEARNING_RATE)
        self.rng = np.random.default_rng(seed=42)

        # Training state
        self.training = False
        self.paused = False
        self.evaluation = False
        self.eval_greedy = True
        self.episode = 0
        self.episode_rewards = []
        self.successes = []
        self.route_lengths = []
        self.last_policy_loss = 0.0
        self.last_entropy = 0.0
        self.last_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.last_action = 0
        self.last_action_label = "-"
        self.reward_history = collections.deque(maxlen=50)
        self.collision_history = collections.deque(maxlen=50)

        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.input_history: List[np.ndarray] = []

        # User control of the agent
        self.user_control = False
        self._arrow_keys = [False, False, False, False]  # up, right, down, left

        # Camera & UI
        self.camera = Camera(view_w=WINDOW_W - DASHBOARD_W, view_h=WINDOW_H)
        self.show_minimap = True
        self.show_paths = True
        self.dragging = False
        self.drag_start = (0, 0)

        # Map editing
        self.edit_mode = False
        self.edit_brush = "wall"
        self._edit_dragging = False
        self._edit_last_cell = None

        # Dashboard scroll
        self.dash_scroll = 0.0
        self.dash_content_h = 0
        self.dash_scroll_speed = 40.0

        # Initial map
        self._install_generated_map(width=30, height=20,
                                    seed=random.randint(0, 2**31 - 1),
                                    save=False)

        # UI widgets
        self.input_width  = TextInput(str(self.env.width))
        self.input_height = TextInput(str(self.env.height))
        self.input_seed   = TextInput("")
        self.saved_maps = list_saved_maps()
        self.map_dropdown = Dropdown(
            self.saved_maps or ["(no saved maps)"], self.font_s
        )

        self._hit_buttons = []
        self._content_cursor = 0

        # CSV log
        self.log_file = LOGS_DIR / "training.csv"
        if not self.log_file.exists():
            with self.log_file.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "steps", "reward", "success", "loss", "entropy"]
                )

        # Simulation speed
        self.speed = 1
        self.step_accum = 0.0

        # Toast
        self.toast = ""
        self.toast_until = 0.0

    # ------------------------------------------------------------------ #
    # Map install helpers
    # ------------------------------------------------------------------ #
    def _new_path_manager(self, map_id: str) -> PathManager:
        pm = PathManager()
        pm.reset_for_map(map_id)
        return pm

    def _install_generated_map(self, width, height, seed,
                               save=True, wall_density=0.25):
        try:
            gen = MazeGenerator(seed=seed)
            cfg = gen.generate(width=width, height=height,
                               wall_density=wall_density, min_routes=1)
        except Exception as e:
            self.flash(f"Map generation failed: {e}")
            return

        self.generator = gen
        self.map_config = cfg
        self.map_id = f"gen_{cfg['seed']}_{int(time.time()) % 100000}"

        if save:
            try:
                save_map(cfg, self.map_id)
                self.saved_maps = list_saved_maps()
                self.map_dropdown.refresh(self.saved_maps or ["(no saved maps)"])
            except Exception as e:
                self.flash(f"Save failed: {e}")

        self.env = MazeEnv(cfg)
        self.path_manager = self._new_path_manager(self.map_id)
        self._reset_episode_buffers()
        self._center_camera_on_agent()

    def new_random_map(self):
        self._install_generated_map(
            width=random.randint(15, 40),
            height=random.randint(10, 30),
            seed=random.randint(0, 2**31 - 1),
            save=True,
        )
        self.flash(f"New random map '{self.map_id}'")

    def new_custom_map(self):
        try:
            w = int(self.input_width.text)
            h = int(self.input_height.text)
            if w < 3: w = 3
            if h < 3: h = 3
        except Exception:
            self.flash("Width and height must be integers")
            return
        seed_txt = self.input_seed.text.strip()
        try:
            seed = int(seed_txt) if seed_txt else random.randint(0, 2**31 - 1)
        except Exception:
            self.flash("Seed must be an integer")
            return

        grid = np.zeros((h, w), dtype=np.uint8)
        start = (0, 0)
        goal = (h - 1, w - 1)
        gen = MazeGenerator(seed=seed)
        d = gen._bfs_distance(grid, start, goal)

        cfg = {
            "grid": grid,
            "start": start,
            "goal": goal,
            "seed": seed,
            "params": {"width": w, "height": h,
                       "wall_density": 0.0, "min_routes": 1,
                       "algorithm": "custom"},
            "shortest_distance": int(d) if d is not None else -1,
            "route_count": 1,
        }
        self.generator = gen
        self.map_config = cfg
        self.map_id = f"custom_{seed}_{int(time.time()) % 100000}"
        self.env = MazeEnv(cfg)
        self.path_manager = self._new_path_manager(self.map_id)
        self._reset_episode_buffers()
        self._center_camera_on_agent()

        self.edit_mode = True
        self.edit_brush = "wall"
        self.flash("Custom map — paint walls, set start/goal, then Save")

    def load_saved_map(self, name):
        try:
            cfg = load_map(name)
        except Exception as e:
            self.flash(f"Load failed: {e}")
            return
        self.map_config = cfg
        self.map_id = cfg.get("id", name)
        self.env = MazeEnv(cfg)
        self.path_manager = self._new_path_manager(self.map_id)
        self._reset_episode_buffers()
        self._center_camera_on_agent()
        self.flash(f"Loaded map '{name}'")

    def save_current_map(self):
        try:
            if self.edit_mode:
                self.env.refresh_distances()
            cfg = self.map_config_from_env()
            save_map(cfg, self.map_id)
            self.saved_maps = list_saved_maps()
            self.map_dropdown.refresh(self.saved_maps or ["(no saved maps)"])
            self.flash(f"Saved map '{self.map_id}'")
        except Exception as e:
            self.flash(f"Save failed: {e}")

    def map_config_from_env(self):
        return {
            "grid": self.env.grid.copy(),
            "start": tuple(self.env.start),
            "goal": tuple(self.env.goal),
            "seed": int(getattr(self.env, "seed", 0)),
            "params": {
                "width": int(self.env.width),
                "height": int(self.env.height),
                "wall_density": 0.0,
                "min_routes": 1,
                "algorithm": "custom",
            },
            "shortest_distance": int(self.env.shortest_distance),
        }

    def _center_camera_on_agent(self):
        self.camera.center_on(
            self.env.agent_c * CELL + CELL / 2,
            self.env.agent_r * CELL + CELL / 2,
        )

    # ------------------------------------------------------------------ #
    # Episode control
    # ------------------------------------------------------------------ #
    def _reset_episode_buffers(self):
        self.state = self.env.reset()
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.input_history = [self.state.copy()]
        self.last_action_label = "-"
        self.episode_success = False
        self._arrow_keys = [False, False, False, False]

    def _on_reset_button(self):
        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
            lr = LEARNING_RATE
            self.model = rl_model.create_policy_model(
                state_dim=STATE_DIM, learning_rate=lr)
            self.optimizer = rl_model.create_optimizer(lr)
            self.reward_history.clear()
            self.episode_rewards.clear()
            self.successes.clear()
            self.route_lengths.clear()
            self.last_policy_loss = 0.0
            self.last_entropy = 0.0
            self.flash("Model re-initialised")
        else:
            self.flash("Episode reset")
        self._reset_episode_buffers()

    def start_training(self):
        self.training = True
        self.evaluation = False
        self.paused = False
        self.edit_mode = False
        self._reset_episode_buffers()
        self.flash("Training started")

    def stop_training(self):
        self.training = False
        self.flash("Training stopped")

    def toggle_eval(self):
        self.evaluation = not self.evaluation
        self.training = False
        self._reset_episode_buffers()
        self.flash(("Evaluation" if self.evaluation else "Idle") + " mode")

    def toggle_user_control(self):
        self.user_control = not self.user_control
        self._arrow_keys = [False, False, False, False]
        self.flash(
            f"User control {'ON' if self.user_control else 'OFF'}"
            + (" — use arrow keys to move" if self.user_control else "")
        )

    def _action_from_user(self) -> Optional[int]:
        """Map the held arrow keys to an action.

        Return ``None`` if no arrow key was pressed this frame, so the caller
        can skip stepping the environment.  The user-facing keys map onto the
        relative-action space as follows:

            Up    -> Forward
            Right -> Right
            Down  -> Backward
            Left  -> Left
        """
        up, right, down, left = self._arrow_keys
        if up:    return 0
        if right: return 1
        if down:  return 2
        if left:  return 3
        return None

    def step_env(self):
        # --------------------------------------------------------------- #
        # Choose the action.  Priority:
        #   1. User control (arrow keys) — wins over everything else
        #   2. Training mode    (stochastic)
        #   3. Evaluation mode  (greedy or stochastic)
        #   4. Idle              (greedy)
        # --------------------------------------------------------------- #
        if self.user_control:
            action = self._action_from_user()
            if action is None:
                return False  # nothing to do this frame
            # Refresh probabilities for display purposes so the dashboard
            # shows what the policy *would* have done at this state.
            self.last_probs = rl_model.action_probabilities(self.model, self.state)
        else:
            self.last_probs = rl_model.action_probabilities(self.model, self.state)
            if self.training:
                action = rl_model.sample_action(self.last_probs, self.rng)
            elif self.evaluation:
                action = (rl_model.greedy_action(self.last_probs)
                          if self.eval_greedy
                          else rl_model.sample_action(self.last_probs, self.rng))
            else:
                action = rl_model.greedy_action(self.last_probs)

        self.last_action = action
        self.last_action_label = ["Fwd", "Right", "Bwd", "Left"][action]
        if self.user_control:
            self.last_action_label += " (user)"

        # The record-pipeline is *identical* whether the action came from the
        # policy or from the user: states, actions, rewards, and gradients
        # all flow through the same code.  This is the whole point of the
        # feature — the user is effectively providing expert demonstrations
        # to the REINFORCE update.
        self.ep_states.append(self.state.copy())
        self.ep_actions.append(action)

        next_state, reward, done, _ = self.env.step(action)
        self.ep_rewards.append(float(reward))
        self.state = next_state
        self.input_history.append(self.state.copy())

        if done:
            self.episode += 1
            success = bool(self.env.success)
            record = EpisodeRecord(
                episode=self.episode,
                steps=self.env.steps,
                total_reward=self.env.total_reward,
                success=success,
                path=list(self.env.current_path),
                actions=list(self.ep_actions),
                rewards=list(self.ep_rewards),
            )
            if self.training:
                loss, ent, _ = rl_model.train_policy_step(
                    self.model, self.optimizer,
                    self.ep_states, self.ep_actions, self.ep_rewards,
                    gamma=GAMMA, entropy_coef=ENTROPY_COEF,
                )
                self.last_policy_loss = loss
                self.last_entropy = ent
                record.policy_loss = loss
                record.entropy = ent
                self.path_manager.record_episode(record)
                self.reward_history.append(self.env.total_reward)
                self.collision_history.append(self.env.collisions)
                self.episode_rewards.append(self.env.total_reward)
                self.successes.append(success)
                if success:
                    self.route_lengths.append(self.env.steps)
                self._log_episode(record)
                if self.episode % 10 == 0:
                    self.path_manager.save_history()
            elif self.evaluation:
                self.path_manager.record_episode(record)
            self.episode_success = success
            self._reset_episode_buffers()
            return True
        return False

    def _log_episode(self, rec):
        try:
            with self.log_file.open("a", newline="") as f:
                csv.writer(f).writerow([
                    rec.episode, rec.steps, f"{rec.total_reward:.3f}",
                    int(rec.success), f"{rec.policy_loss:.5f}",
                    f"{rec.entropy:.5f}",
                ])
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            running = self._handle_events(dt)
            self._update(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()
        try: self.path_manager.save_history()
        except Exception: pass

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def _handle_events(self, dt):
        # Track held arrow keys each frame for user-control movement.
        keys = pygame.key.get_pressed()
        if self.user_control:
            self._arrow_keys = [
                keys[pygame.K_UP],
                keys[pygame.K_RIGHT],
                keys[pygame.K_DOWN],
                keys[pygame.K_LEFT],
            ]

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False

            # ---------------- Widget events FIRST -------------------- #
            widget_used = False
            widget_used |= self.input_width.handle_event(e)
            widget_used |= self.input_height.handle_event(e)
            widget_used |= self.input_seed.handle_event(e)
            widget_used |= self.map_dropdown.handle_event(e)
            if widget_used:
                continue

            if e.type == pygame.KEYDOWN:
                k = e.key
                any_text_active = (self.input_width.active
                                   or self.input_height.active
                                   or self.input_seed.active)
                if any_text_active:
                    continue

                # Arrow keys are reserved for user-control — swallow here.
                if k in (pygame.K_UP, pygame.K_RIGHT,
                         pygame.K_DOWN, pygame.K_LEFT):
                    continue

                if k == pygame.K_ESCAPE:
                    if self.edit_mode:
                        self.edit_mode = False
                        self.env.refresh_distances()
                        self.flash("Exited edit mode")
                        continue
                    return False
                elif k == pygame.K_f:
                    self.camera.follow = not self.camera.follow
                    self.flash(f"Follow = {self.camera.follow}")
                elif k == pygame.K_SPACE:
                    self._center_camera_on_agent()
                    self.camera.follow = True
                elif k == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                elif k == pygame.K_v:
                    self.show_paths = not self.show_paths
                elif k == pygame.K_p:
                    self.paused = not self.paused
                    self.flash("Paused" if self.paused else "Resumed")
                elif k == pygame.K_t:
                    self.stop_training() if self.training else self.start_training()
                elif k == pygame.K_e:
                    self.toggle_eval()
                elif k == pygame.K_n:
                    self._reset_episode_buffers()
                    self.flash("Episode reset")
                elif k == pygame.K_g:
                    self.new_random_map()
                elif k == pygame.K_k:
                    p = save_weights(self.model, self.optimizer)
                    self.flash(f"Model saved: {p.name}")
                elif k == pygame.K_l:
                    ok = load_weights(self.model, self.optimizer)
                    self.flash("Model loaded" if ok else "No saved model found")
                elif k == pygame.K_x:
                    self.toggle_user_control()
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3):
                    self.speed = {pygame.K_1: 1, pygame.K_2: 4, pygame.K_3: 16}[k]
                    self.flash(f"Speed {self.speed}x")

            # ---------------- Wheel -------------------------------- #
            if e.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx > WINDOW_W - DASHBOARD_W:
                    self.dash_scroll -= e.y * self.dash_scroll_speed
                else:
                    factor = 1.12 if e.y > 0 else 1 / 1.12
                    self.camera.zoom_at(factor, mx, my)

            # ---------------- Mouse down --------------------------- #
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = e.pos

                if mx > WINDOW_W - DASHBOARD_W and e.button == 1:
                    for rect, cb in self._hit_buttons:
                        if rect.collidepoint(e.pos):
                            cb()
                            break
                    continue

                if self.show_minimap and self._minimap_rect().collidepoint(e.pos):
                    if e.button == 1:
                        self._minimap_click(e.pos)
                    continue

                if self.edit_mode and mx <= WINDOW_W - DASHBOARD_W:
                    r, c = self._screen_to_cell(mx, my)
                    if 0 <= r < self.env.height and 0 <= c < self.env.width:
                        if e.button == 1:
                            self._apply_brush(r, c)
                            self._edit_dragging = True
                            self._edit_last_cell = (r, c)
                        elif e.button == 3:
                            self.env.grid[r, c] = 0
                    continue

                if e.button == 1:
                    self.dragging = True
                    self.drag_start = e.pos

            if e.type == pygame.MOUSEBUTTONUP and e.button in (1, 3):
                self.dragging = False
                self._edit_dragging = False
                self._edit_last_cell = None

            if e.type == pygame.MOUSEMOTION:
                if self.edit_mode and self._edit_dragging:
                    r, c = self._screen_to_cell(*e.pos)
                    if (0 <= r < self.env.height
                            and 0 <= c < self.env.width
                            and (r, c) != self._edit_last_cell):
                        self._apply_brush(r, c)
                        self._edit_last_cell = (r, c)
                elif self.dragging:
                    mx, my = e.pos
                    if mx <= WINDOW_W - DASHBOARD_W:
                        dx = mx - self.drag_start[0]
                        dy = my - self.drag_start[1]
                        self.camera.pan(-dx, -dy)
                        self.drag_start = e.pos
                        self.camera.follow = False

        # WASD panning — camera-only control now (arrows belong to the agent).
        pan = np.array([0.0, 0.0], dtype=np.float32)
        if keys[pygame.K_a]: pan[0] -= 1
        if keys[pygame.K_d]: pan[0] += 1
        if keys[pygame.K_w]: pan[1] -= 1
        if keys[pygame.K_s]: pan[1] += 1
        if pan.any():
            self.camera.pan(pan[0] * PAN_SPEED * dt, pan[1] * PAN_SPEED * dt)
            self.camera.follow = False
        return True

    # ------------------------------------------------------------------ #
    # Editing helpers
    # ------------------------------------------------------------------ #
    def _screen_to_cell(self, sx, sy):
        wx, wy = self.camera.screen_to_world(sx, sy)
        return int(wy // CELL), int(wx // CELL)

    def _apply_brush(self, r, c):
        if (r, c) == self.env.start or (r, c) == self.env.goal:
            return
        if self.edit_brush == "wall":
            self.env.grid[r, c] = 1
        elif self.edit_brush == "erase":
            self.env.grid[r, c] = 0
        elif self.edit_brush == "start":
            self.env.start = (r, c)
            self.env.grid[r, c] = 0
        elif self.edit_brush == "goal":
            self.env.goal = (r, c)
            self.env.grid[r, c] = 0

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def _update(self, dt):
        if self.camera.follow:
            self.camera.smooth_follow(self.env.agent_c * CELL + CELL / 2,
                                      self.env.agent_r * CELL + CELL / 2)
        self.camera.clamp(self.env.width, self.env.height, CELL)

        if not self.paused and not self.edit_mode:
            if self.user_control:
                # User control is frame-rate based — one action per frame
                # maximum so holding a key doesn't teleport the agent.
                self.step_env()
            elif self.training:
                self.step_accum += dt * 20.0 * self.speed
                n = int(self.step_accum); self.step_accum -= n
                for _ in range(min(n, 64)):
                    self.step_env()
            else:
                self.step_accum += dt * 20.0 * self.speed
                n = int(self.step_accum); self.step_accum -= n
                for _ in range(min(n, 32)):
                    self.step_env()

        max_scroll = max(0, self.dash_content_h - WINDOW_H + 20)
        self.dash_scroll = max(0, min(max_scroll, self.dash_scroll))

    # ------------------------------------------------------------------ #
    # Draw
    # ------------------------------------------------------------------ #
    def _draw(self):
        self.screen.fill(COLORS["bg"])
        self._draw_viewport()
        self._draw_dashboard()
        if self.show_minimap:
            self._draw_minimap()
        self.map_dropdown.draw_list(self.screen)
        self._draw_toast()

    def _cell_rect(self, r, c, z):
        sx, sy = self.camera.world_to_screen(c * CELL, r * CELL)
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return pygame.Rect(0, 0, 1, 1)
        size = max(1, int(CELL * z))
        sx = max(-100000, min(100000, int(sx)))
        sy = max(-100000, min(100000, int(sy)))
        return pygame.Rect(sx, sy, size, size)

    def _draw_viewport(self):
        z = self.camera.zoom
        c0, r0, c1, r1 = self.camera.visible_world_rect(CELL)
        c0 = max(c0, 0); r0 = max(r0, 0)
        c1 = min(c1, self.env.width - 1)
        r1 = min(r1, self.env.height - 1)

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rect = self._cell_rect(r, c, z)
                color = COLORS["wall"] if self.env.grid[r, c] == 1 else COLORS["cell"]
                pygame.draw.rect(self.screen, color, rect)

        for (r, c) in self.env.visited:
            if r0 <= r <= r1 and c0 <= c <= c1:
                rect = self._cell_rect(r, c, z)
                s = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                s.fill((*COLORS["visited"], 90))
                self.screen.blit(s, rect.topleft)

        if self.show_paths:
            self._draw_polyline(self.env.current_path, COLORS["current_path"], z, 2)
            if self.path_manager.best_path:
                self._draw_polyline(self.path_manager.best_path,
                                    COLORS["best_path"], z, 3)

        self._draw_marker(self.env.start, COLORS["start"], z)
        self._draw_marker(self.env.goal, COLORS["goal"], z)
        self._draw_agent(z)

        pygame.draw.rect(self.screen, COLORS["dash_line"],
                         (0, 0, WINDOW_W - DASHBOARD_W, WINDOW_H), 1)

        if self.edit_mode:
            pygame.draw.rect(self.screen, COLORS["good"],
                             (0, 0, WINDOW_W - DASHBOARD_W, WINDOW_H), 2)
            hint = self.font_m.render(
                f"EDIT MODE  •  brush={self.edit_brush}  •  "
                f"LMB=paint  RMB=erase  ESC=exit",
                True, COLORS["good"])
            self.screen.blit(hint, (10, 10))
        elif self.user_control:
            pygame.draw.rect(self.screen, COLORS["agent_user"],
                             (0, 0, WINDOW_W - DASHBOARD_W, WINDOW_H), 2)
            hint = self.font_m.render(
                "USER CONTROL  •  arrow keys to move  •  X to release",
                True, COLORS["agent_user"])
            self.screen.blit(hint, (10, 10))

    def _draw_marker(self, pos, color, z):
        r, c = pos
        rect = self._cell_rect(r, c, z)
        pad = max(2, rect.w // 6)
        pygame.draw.rect(self.screen, color,
                        rect.inflate(-2 * pad, -2 * pad), border_radius=4)

    def _draw_agent(self, z):
        rect = self._cell_rect(self.env.agent_r, self.env.agent_c, z)
        cx, cy = rect.center
        radius = max(3, rect.w // 3)
        agent_color = COLORS["agent_user"] if self.user_control else COLORS["agent"]
        pygame.draw.circle(self.screen, agent_color, (cx, cy), radius)
        dr, dc = DIR_VECTORS[self.env.orientation]
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy),
                        (cx + dc * radius, cy + dr * radius), 2)

    def _draw_polyline(self, path, color, z, width=2):
        if len(path) < 2:
            return
        pts = []
        for (r, c) in path:
            sx, sy = self.camera.world_to_screen(
                c * CELL + CELL / 2,
                r * CELL + CELL / 2,
            )
            if not (math.isfinite(sx) and math.isfinite(sy)):
                continue
            sx = max(-100000.0, min(100000.0, sx))
            sy = max(-100000.0, min(100000.0, sy))
            pts.append((int(round(sx)), int(round(sy))))

        if len(pts) < 2:
            return
        try:
            pygame.draw.lines(self.screen, color, False, pts, max(1, width))
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------ #
    # Dashboard
    # ------------------------------------------------------------------ #
    def _draw_dashboard(self):
        x0 = WINDOW_W - DASHBOARD_W
        pygame.draw.rect(self.screen, COLORS["dash_bg"],
                         (x0, 0, DASHBOARD_W, WINDOW_H))
        pygame.draw.line(self.screen, COLORS["dash_line"],
                         (x0, 0), (x0, WINDOW_H), 2)

        content = pygame.Surface((DASHBOARD_W, 20000), pygame.SRCALPHA)
        content.fill((0, 0, 0, 0))

        self._build_dashboard_content(content, x_off=14, y_off=14)
        self.dash_content_h = self._content_cursor

        max_scroll = max(0, self.dash_content_h - WINDOW_H + 20)
        self.dash_scroll = max(0, min(max_scroll, self.dash_scroll))

        self.screen.blit(
            content, (x0, 0),
            area=pygame.Rect(0, int(self.dash_scroll), DASHBOARD_W, WINDOW_H)
        )

        if self.dash_content_h > WINDOW_H:
            bar_w = 8
            bar_x = WINDOW_W - bar_w
            thumb_h = max(30, int(WINDOW_H * (WINDOW_H / self.dash_content_h)))
            max_y = WINDOW_H - thumb_h
            frac = 0.0 if max_scroll == 0 else (self.dash_scroll / max_scroll)
            thumb_y = int(frac * max_y)
            pygame.draw.rect(self.screen, COLORS["scrollbar_bg"],
                             (bar_x, 0, bar_w, WINDOW_H))
            pygame.draw.rect(self.screen, COLORS["scrollbar"],
                             (bar_x, thumb_y, bar_w, thumb_h), border_radius=3)

    def _build_dashboard_content(self, surf, x_off, y_off):
        x = x_off
        y = y_off
        self._hit_buttons = []

        def text(s, color=None, font=None):
            nonlocal y
            f = font or self.font_s
            s2 = f.render(s, True, color or COLORS["text"])
            surf.blit(s2, (x, y))
            y += s2.get_height() + 2

        def line():
            nonlocal y
            pygame.draw.line(surf, COLORS["dash_line"], (x, y),
                             (x + DASHBOARD_W - 28, y))
            y += 8

        def register_button(rect_local, cb):
            scr = pygame.Rect(
                rect_local.x + (WINDOW_W - DASHBOARD_W),
                rect_local.y - int(self.dash_scroll),
                rect_local.w, rect_local.h,
            )
            self._hit_buttons.append((scr, cb))

        def button(label, cb, h=26):
            nonlocal y
            r = pygame.Rect(x, y, DASHBOARD_W - 28, h)
            hover = r.collidepoint(
                pygame.mouse.get_pos()[0] - (WINDOW_W - DASHBOARD_W),
                pygame.mouse.get_pos()[1] + int(self.dash_scroll),
            )
            base = COLORS["button_hover"] if hover else COLORS["button"]
            pygame.draw.rect(surf, base, r, border_radius=5)
            pygame.draw.rect(surf, COLORS["accent"], r, 1, border_radius=5)
            lbl = self.font_s.render(label, True, COLORS["text"])
            surf.blit(lbl, (r.centerx - lbl.get_width() // 2,
                            r.centery - lbl.get_height() // 2))
            register_button(r, cb)
            y += h + 6

        def input_row(label, widget):
            nonlocal y
            surf.blit(self.font_s.render(label, True, COLORS["text"]), (x, y + 3))
            local = pygame.Rect(x + 90, y, DASHBOARD_W - 120, 22)
            widget.draw(surf, self.font_s, local)
            widget.rect = pygame.Rect(
                local.x + (WINDOW_W - DASHBOARD_W),
                local.y - int(self.dash_scroll),
                local.w, local.h,
            )
            y += 26

        # header
        text("MAZE RL NAVIGATION", COLORS["accent"], self.font_l)
        text(f"Map: {self.env.width}x{self.env.height}  id={self.map_id}",
             COLORS["text_dim"])
        text(f"Shortest: {self.env.shortest_distance}  Routes>= "
             f"{self.map_config.get('params', {}).get('min_routes', 1)}",
             COLORS["text_dim"])
        y += 4; line()

        # mode
        if self.training:
            mode, mcol = "TRAINING", COLORS["good"]
        elif self.evaluation:
            mode, mcol = "EVALUATION", COLORS["goal"]
        else:
            mode, mcol = "IDLE", COLORS["text_dim"]
        if self.paused:
            mode += " (PAUSED)"; mcol = COLORS["bad"]
        if self.edit_mode:
            mode += "  [EDIT]"
        if self.user_control:
            mode += "  [USER]"; mcol = COLORS["agent_user"]
        text(mode, mcol, self.font_m)
        text(f"Speed: {self.speed}x  Follow: {self.camera.follow}  "
             f"Zoom: {self.camera.zoom:.2f}", COLORS["text_dim"])
        y += 4; line()

        # stats
        text(f"Episode     : {self.episode}", COLORS["text"], self.font_m)
        text(f"Step        : {self.env.steps} / {MAX_STEPS}")
        text(f"Ep reward   : {self.env.total_reward:+.2f}")
        text(f"Collisions  : {self.env.collisions}")
        text(f"Last action : {self.last_action_label}")
        text(f"Policy loss : {self.last_policy_loss:.4f}")
        text(f"Entropy     : {self.last_entropy:.4f}")
        text(f"Input hist. : {len(self.input_history)} obs", COLORS["text_dim"])

        sr = (f"{(sum(self.successes)/len(self.successes))*100:.1f}%"
              if self.successes else "-")
        mar = (f"{sum(self.reward_history)/len(self.reward_history):+.2f}"
               if self.reward_history else "-")
        avg_len = (sum(self.route_lengths)/len(self.route_lengths)
                   if self.route_lengths else 0.0)
        text(f"Successes   : {sum(self.successes)} / {len(self.successes)}  ({sr})")
        text(f"Avg reward  : {mar}")
        text(f"Avg route   : {avg_len:.1f}")

        if self.path_manager.best_path:
            text(f"Best path   : {self.path_manager.best_length} steps "
                 f"(ep {self.path_manager.best_episode})", COLORS["best_path"])
        else:
            text("Best path   : none", COLORS["text_dim"])
        y += 4; line()

        # probabilities
        text("Softmax probabilities", COLORS["text"], self.font_m)
        labels = ["Forward", "Right  ", "Backward", "Left   "]
        bar_x = x + 84
        bar_w = DASHBOARD_W - 120
        for i, lbl in enumerate(labels):
            p = float(self.last_probs[i])
            col = COLORS["accent"] if i == self.last_action else COLORS["text"]
            surf.blit(self.font_s.render(lbl, True, col), (x, y))
            bx, by, bh = bar_x, y + 2, 12
            pygame.draw.rect(surf, COLORS["grid"], (bx, by, bar_w, bh),
                             border_radius=3)
            fill_w = int(bar_w * p)
            bcol = COLORS["accent"] if i == self.last_action else COLORS["text_dim"]
            if fill_w > 0:
                pygame.draw.rect(surf, bcol, (bx, by, fill_w, bh), border_radius=3)
            surf.blit(self.font_s.render(f"{p*100:5.1f}%", True, COLORS["text"]),
                      (bx + bar_w + 4, by - 2))
            y += 18
        y += 4; line()

        # reward graph
        text("Reward graph (last 50)", COLORS["text"], self.font_m)
        gr = pygame.Rect(x, y, DASHBOARD_W - 28, 110)
        pygame.draw.rect(surf, COLORS["grid"], gr, border_radius=4)
        if len(self.reward_history) >= 2:
            hist = list(self.reward_history)
            lo, hi = min(hist), max(hist)
            if hi - lo < 1e-3: hi = lo + 1.0
            pts = []
            n = len(hist)
            for i, v in enumerate(hist):
                px = gr.x + 2 + (i / (n - 1)) * (gr.w - 4)
                py = gr.bottom - 2 - ((v - lo) / (hi - lo)) * (gr.h - 4)
                pts.append((px, py))
            pygame.draw.lines(surf, COLORS["good"], False, pts, 2)
            if lo <= 0 <= hi:
                zy = gr.bottom - 2 - ((0 - lo) / (hi - lo)) * (gr.h - 4)
                pygame.draw.line(surf, COLORS["dash_line"],
                                 (gr.x, zy), (gr.right, zy), 1)
            surf.blit(self.font_s.render(f"{hi:+.0f}", True, COLORS["text_dim"]),
                      (gr.x + 4, gr.y + 2))
            surf.blit(self.font_s.render(f"{lo:+.0f}", True, COLORS["text_dim"]),
                      (gr.x + 4, gr.bottom - 16))
        y += gr.h + 8; line()

        # user control toggle
        text("Agent Control", COLORS["text"], self.font_m)
        uc_label = (f"User Control: {'ON  (arrow keys)' if self.user_control else 'OFF  (policy)'}")
        uc_col = COLORS["agent_user"] if self.user_control else COLORS["text_dim"]
        text(uc_label, uc_col)
        button(f"Toggle User Control  [X]", self.toggle_user_control)
        y += 4; line()

        # custom map
        text("Custom Map", COLORS["text"], self.font_m)
        input_row("Width :", self.input_width)
        input_row("Height:", self.input_height)
        input_row("Seed  :", self.input_seed)

        button("Create Custom Map", self.new_custom_map)

        # saved maps
        text("Saved Maps", COLORS["text"], self.font_m)
        local = pygame.Rect(x, y, DASHBOARD_W - 28, 22)
        self.map_dropdown.draw_box(surf, local)
        self.map_dropdown.box_rect = pygame.Rect(
            local.x + (WINDOW_W - DASHBOARD_W),
            local.y - int(self.dash_scroll),
            local.w, local.h,
        )
        y += 30

        button("Load Selected Map", lambda: self.load_saved_map(
            self.map_dropdown.selected or ""))

        def refresh():
            self.saved_maps = list_saved_maps()
            self.map_dropdown.refresh(self.saved_maps or ["(no saved maps)"])

        button("Refresh Saved Maps", refresh)
        button("Save Current Map", self.save_current_map)
        y += 4; line()

        # action buttons
        button("Reset Episode  [N]", self._on_reset_button)
        button("New Random Map  [G]", self.new_random_map)
        y += 4; line()

        # edit-mode panel
        text("Map Editor", COLORS["text"], self.font_m)
        text(f"Mode: {'ON' if self.edit_mode else 'OFF'}", COLORS["text_dim"])

        def toggle_edit():
            self.edit_mode = not self.edit_mode
            if not self.edit_mode:
                self.env.refresh_distances()
            self.flash(f"Map editor {'enabled' if self.edit_mode else 'disabled'}")

        button(f"Edit Mode: {'ON' if self.edit_mode else 'OFF'}", toggle_edit)

        brush_labels = [("wall", "Paint Wall"), ("erase", "Erase Cell"),
                        ("start", "Move Start"), ("goal", "Move Goal")]
        bw = (DASHBOARD_W - 28 - 4) // 2
        bh = 24
        for i, (bkey, bname) in enumerate(brush_labels):
            col = i % 2; row = i // 2
            r = pygame.Rect(x + col * (bw + 4), y + row * (bh + 4), bw, bh)
            active = (self.edit_brush == bkey)

            def mk(k):
                def setter():
                    self.edit_brush = k
                    self.edit_mode = True
                return setter
            cb = mk(bkey)
            hover = r.collidepoint(
                pygame.mouse.get_pos()[0] - (WINDOW_W - DASHBOARD_W),
                pygame.mouse.get_pos()[1] + int(self.dash_scroll),
            )
            base = (COLORS["accent"] if active
                    else COLORS["button_hover"] if hover
                    else COLORS["button"])
            pygame.draw.rect(surf, base, r, border_radius=5)
            pygame.draw.rect(surf, COLORS["accent"], r, 1, border_radius=5)
            lbl = self.font_s.render(bname, True, COLORS["text"])
            surf.blit(lbl, (r.centerx - lbl.get_width() // 2,
                            r.centery - lbl.get_height() // 2))
            register_button(r, cb)
        y += 2 * (bh + 4) + 4

        button("Save Edited Map", self.save_current_map)
        y += 4; line()

        # help
        text("Controls", COLORS["text"], self.font_m)
        for l in [
            "T  start/stop training",
            "E  evaluation mode   P pause",
            "G  new random map    N reset ep",
            "K  save model        L load model",
            "F  follow agent      SPACE centre",
            "M  minimap   V paths  1/2/3 speed",
            "X  toggle user control",
            "Arrow keys = move agent (user ctl)",
            "WASD  or drag = pan camera",
            "Mouse wheel = zoom (viewport)",
            "Mouse wheel = scroll (dashboard)",
            "",
            "Custom map workflow:",
            "1. Set width / height / seed",
            "2. Create Custom Map",
            "3. Paint walls (LMB)",
            "4. Place Start / Goal",
            "5. Save Current Map",
            "",
            "Colours: edit COLORS dict",
            "at the top of Game.py.",
        ]:
            text(l, COLORS["text_dim"])

        self._content_cursor = y

    # ------------------------------------------------------------------ #
    # Minimap
    # ------------------------------------------------------------------ #
    def _minimap_rect(self):
        size = 200
        margin = 10
        return pygame.Rect(margin, WINDOW_H - size - margin, size, size)

    def _draw_minimap(self):
        rect = self._minimap_rect()
        surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        surf.fill((10, 12, 18, 200))
        pygame.draw.rect(surf, COLORS["dash_line"], surf.get_rect(), 1)

        h, w = self.env.height, self.env.width
        scale = min((rect.w - 8) / w, (rect.h - 8) / h)
        off_x = (rect.w - w * scale) / 2
        off_y = (rect.h - h * scale) / 2

        step = max(1, int(1.0 / scale)) if scale < 1 else 1
        if w * h <= 250_000:
            for r in range(0, h, step):
                for c in range(0, w, step):
                    if self.env.grid[r, c] == 1:
                        col = COLORS["wall"]
                    elif (r, c) in self.env.visited:
                        col = COLORS["visited"]
                    else:
                        col = COLORS["cell"]
                    pygame.draw.rect(
                        surf, col,
                        (off_x + c * scale, off_y + r * scale,
                         max(1, scale * step), max(1, scale * step)))

        for pos, col in ((self.env.start, COLORS["start"]),
                         (self.env.goal, COLORS["goal"])):
            r, c = pos
            pygame.draw.circle(
                surf, col,
                (int(off_x + (c + 0.5) * scale),
                 int(off_y + (r + 0.5) * scale)), 3)

        agent_col = COLORS["agent_user"] if self.user_control else COLORS["agent"]
        pygame.draw.circle(
            surf, agent_col,
            (int(off_x + (self.env.agent_c + 0.5) * scale),
             int(off_y + (self.env.agent_r + 0.5) * scale)), 3)

        c0, r0, c1, r1 = self.camera.visible_world_rect(CELL)
        vr = pygame.Rect(
            int(off_x + max(0, c0) * scale),
            int(off_y + max(0, r0) * scale),
            int((min(w, c1) - max(0, c0)) * scale),
            int((min(h, r1) - max(0, r0)) * scale))
        pygame.draw.rect(surf, COLORS["accent"], vr, 1)

        self.screen.blit(surf, rect.topleft)

    def _minimap_click(self, pos):
        rect = self._minimap_rect()
        h, w = self.env.height, self.env.width
        scale = min((rect.w - 8) / w, (rect.h - 8) / h)
        off_x = (rect.w - w * scale) / 2
        off_y = (rect.h - h * scale) / 2
        lx = pos[0] - rect.x - off_x
        ly = pos[1] - rect.y - off_y
        if scale <= 0: return
        self.camera.center_on((lx / scale) * CELL, (ly / scale) * CELL)
        self.camera.follow = False

    # ------------------------------------------------------------------ #
    # Toast
    # ------------------------------------------------------------------ #
    def flash(self, msg, seconds=2.5):
        self.toast = msg
        self.toast_until = time.time() + seconds

    def _draw_toast(self):
        if not self.toast or time.time() > self.toast_until:
            return
        surf = self.font_m.render(self.toast, True, (255, 255, 255))
        pad = 8
        bg = pygame.Surface(
            (surf.get_width() + 2 * pad, surf.get_height() + pad),
            pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        vx = (WINDOW_W - DASHBOARD_W - bg.get_width()) // 2
        vy = WINDOW_H - bg.get_height() - 12
        self.screen.blit(bg, (vx, vy))
        self.screen.blit(surf, (vx + pad, vy + pad // 2))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    game = Game()
    try:
        game.run()
    except KeyboardInterrupt:
        pass
    finally:
        try: game.path_manager.save_history()
        except Exception: pass


if __name__ == "__main__":
    main()