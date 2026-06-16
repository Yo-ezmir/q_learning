# Custom FrozenLake environment with live Q-value overlay for demo/visualization purposes.
# Based on gymnasium's built-in frozen_lake.py, extended to render Q-values on top of the map
# plus keyboard shortcuts to control animation speed. Windows/local use only (requires pygame display).

from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

import pygame

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    valid = False
    board = []

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class FrozenLakeQVizEnv(Env):
    """
    FrozenLake with a live Q-value overlay rendered on each tile, for visual demos.
    Same dynamics as gymnasium's FrozenLake-v1 (16 states for the 4x4 map, 4 actions).
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        agent_label: str = "Agent",
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(110 * ncol, 640), min(110 * nrow, 640))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None

        # Additional variables
        self.q_table = None
        self.episode = "---"
        self.pygame_initialized = False
        self.text_padding = 5
        self.agent_label = agent_label  # e.g. "Non-Slippery Agent" / "Slippery Agent"
        self.is_slippery_flag = is_slippery

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.pygame_initialized:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        import sys
                        sys.exit()
                    elif event.key == pygame.K_EQUALS:
                        self.metadata["render_fps"] += 10
                    elif event.key == pygame.K_MINUS:
                        self.metadata["render_fps"] -= 10
                        if self.metadata["render_fps"] <= 0:
                            self.metadata["render_fps"] = 1
                    elif event.key == pygame.K_0:
                        self.metadata["render_fps"] = 0
                    elif event.key == pygame.K_1:
                        self.metadata["render_fps"] = 4
                    elif event.key == pygame.K_9:
                        self.render_mode = None if (self.render_mode == "human") else "human"

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            self.pygame_initialized = True

            self.ui_font = pygame.font.SysFont("Courier", 22)
            self.ui_font_bold = pygame.font.SysFont("Courier", 24, True)
            self.q_font = pygame.font.SysFont("Courier", 11)
            self.q_font_bold = pygame.font.SysFont("Courier", 11, True)

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption(f"Q-Learning FrozenLake — {self.agent_label}")
                self.window_surface = pygame.display.set_mode((960, 600))
                self.display_width, display_height = pygame.display.get_surface().get_size()

                self.grid_size = min(display_height, self.display_width - 340)
                self.window_size = (self.grid_size, self.grid_size)
                self.cell_size = (
                    self.window_size[0] // self.ncol,
                    self.window_size[1] // self.nrow,
                )
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Flat color palette (no image sprites needed)
        ICE_COLOR = (223, 240, 251)        # light icy blue
        ICE_BORDER = (156, 198, 230)       # subtle ice border
        HOLE_COLOR = (40, 40, 40)          # dark circle for holes
        HOLE_RING = (90, 90, 90)
        GOAL_COLOR = (58, 138, 92)         # solid green goal tile
        START_COLOR = (180, 215, 245)      # slightly deeper tint for start tile
        AGENT_COLOR = (220, 90, 40)        # warm orange agent marker
        AGENT_CRACKED_COLOR = (150, 40, 30)

        self.window_surface.fill((255, 255, 255))

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)
                cw, ch = self.cell_size
                cx, cy = pos[0] + cw / 2, pos[1] + ch / 2

                letter = desc[y][x]
                tile_color = START_COLOR if letter == b"S" else (GOAL_COLOR if letter == b"G" else ICE_COLOR)
                pygame.draw.rect(self.window_surface, tile_color, rect)
                pygame.draw.rect(self.window_surface, ICE_BORDER, rect, 1)

                if letter == b"H":
                    radius = min(cw, ch) * 0.28
                    pygame.draw.circle(self.window_surface, HOLE_COLOR, (cx, cy), radius)
                    pygame.draw.circle(self.window_surface, HOLE_RING, (cx, cy), radius, 2)
                elif letter == b"G":
                    flag_w, flag_h = cw * 0.05, ch * 0.35
                    pole_x = cx - cw * 0.08
                    pole_top = cy - ch * 0.22
                    pygame.draw.rect(self.window_surface, (245, 245, 245),
                                      (pole_x, pole_top, flag_w, flag_h))
                    pygame.draw.polygon(self.window_surface, (245, 245, 245), [
                        (pole_x + flag_w, pole_top),
                        (pole_x + flag_w + cw * 0.16, pole_top + flag_h * 0.22),
                        (pole_x + flag_w, pole_top + flag_h * 0.45),
                    ])

                if self.q_table is not None:
                    state = self.nrow * y + x
                    max_q_idx = np.argmax(self.q_table[state])
                    if self.q_table[state][max_q_idx] == 0:
                        max_q_idx = -1

                    q_img = self.q_font.render(".0000", True, (0, 0, 0), (255, 255, 255))
                    q_img_x = q_img.get_width()
                    q_img_y = q_img.get_height()
                    q_pos = [
                        (pos[0] + self.text_padding, pos[1] + self.cell_size[1] / 2),
                        (pos[0] + self.cell_size[0] / 2 - q_img_x / 2,
                         pos[1] + self.cell_size[1] - self.text_padding - q_img_y),
                        (pos[0] + self.cell_size[0] - self.text_padding - q_img_x,
                         pos[1] + self.cell_size[1] / 2),
                        (pos[0] + self.cell_size[0] / 2 - q_img_x / 2, pos[1] + self.text_padding),
                    ]

                    for i in range(4):
                        q = "{:.4f}".format(self.q_table[state][i]).lstrip("0")
                        if max_q_idx == i:
                            q_img = self.q_font_bold.render(q, True, (0, 100, 0), (255, 255, 255))
                        else:
                            q_img = self.q_font.render(q, True, (0, 0, 0), (255, 255, 255))
                        self.window_surface.blit(q_img, q_pos[i])

        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        cw, ch = self.cell_size
        acx, acy = cell_rect[0] + cw / 2, cell_rect[1] + ch / 2

        if desc[bot_row][bot_col] == b"H":
            radius = min(cw, ch) * 0.3
            pygame.draw.circle(self.window_surface, AGENT_CRACKED_COLOR, (acx, acy), radius)
            pygame.draw.line(self.window_surface, (255, 255, 255),
                              (acx - radius * 0.5, acy - radius * 0.5),
                              (acx + radius * 0.5, acy + radius * 0.5), 3)
            pygame.draw.line(self.window_surface, (255, 255, 255),
                              (acx - radius * 0.5, acy + radius * 0.5),
                              (acx + radius * 0.5, acy - radius * 0.5), 3)
        else:
            radius = min(cw, ch) * 0.32
            pygame.draw.circle(self.window_surface, AGENT_COLOR, (acx, acy), radius)
            pygame.draw.circle(self.window_surface, (255, 255, 255), (acx, acy), radius, 2)
            # direction triangle indicating last action: LEFT=0, DOWN=1, RIGHT=2, UP=3
            tip = radius * 0.85
            if last_action == LEFT:
                pts = [(acx - tip, acy), (acx + tip * 0.3, acy - tip * 0.6), (acx + tip * 0.3, acy + tip * 0.6)]
            elif last_action == DOWN:
                pts = [(acx, acy + tip), (acx - tip * 0.6, acy - tip * 0.3), (acx + tip * 0.6, acy - tip * 0.3)]
            elif last_action == RIGHT:
                pts = [(acx + tip, acy), (acx - tip * 0.3, acy - tip * 0.6), (acx - tip * 0.3, acy + tip * 0.6)]
            else:  # UP
                pts = [(acx, acy - tip), (acx - tip * 0.6, acy + tip * 0.3), (acx + tip * 0.6, acy + tip * 0.3)]
            pygame.draw.polygon(self.window_surface, (255, 255, 255), pts)

        if mode == "human":
            panel_x = self.grid_size + self.text_padding * 3

            title_img = self.ui_font_bold.render(self.agent_label, True, (20, 20, 20), (255, 255, 255))
            self.window_surface.blit(title_img, (panel_x, self.text_padding))

            mode_label = "Slippery" if self.is_slippery_flag else "Non-Slippery"
            mode_img = self.ui_font.render(f"Mode: {mode_label}", True, (60, 60, 60), (255, 255, 255))
            self.window_surface.blit(mode_img, (panel_x, self.text_padding + 30))

            fps_img = self.ui_font.render(str(int(self.clock.get_fps())) + " fps", True, (0, 0, 0), (255, 255, 255))
            self.window_surface.blit(fps_img, (panel_x, self.text_padding + 60))

            ep_img = self.ui_font.render("Episode: " + str(self.episode), True, (0, 0, 0), (255, 255, 255))
            self.window_surface.blit(ep_img, (panel_x, self.text_padding + 90))

            text_lines = [
                "",
                "Shortcuts:",
                "1 : Reset speed",
                "0 : Max speed",
                "- : Slow down",
                "= : Speed up",
                "9 : Toggle render",
                "ESC : Quit",
                "",
                "Bold green Q-value =",
                "agent's chosen action",
            ]
            starting_y = self.text_padding + 130
            line_height = 22
            for i, line in enumerate(text_lines):
                line_img = self.ui_font.render(line, True, (40, 40, 40), (255, 255, 255))
                self.window_surface.blit(line_img, (panel_x, starting_y + i * line_height))

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def set_q(self, q_table):
        self.q_table = q_table

    def set_episode(self, episode):
        self.episode = episode