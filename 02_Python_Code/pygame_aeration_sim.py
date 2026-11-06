"""Animated Pygame interface for the existing WastewaterAerationEnv.
Run from project root: python -m pygame_aeration_sim (or run the file directly)
Requires: pygame, numpy, pandas, gymnasium and local config.py/wastewater_env.py.
"""
import math
import random
import sys
from collections import deque
from pathlib import Path

import pygame

from wastewater_env import WastewaterAerationEnv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / '01_Dataset' / 'processed'
DATASETS = {
    'Dry': DATA_DIR / 'dry_influent.csv',
    'Rain': DATA_DIR / 'rain_influent.csv',
    'Storm': DATA_DIR / 'storm_influent.csv',
}
W, H = 1280, 780
FPS = 60
BG = (10, 19, 35)
PANEL = (20, 33, 54)
BORDER = (49, 73, 102)
WHITE = (232, 242, 251)
MUTED = (145, 168, 192)
CYAN = (56, 214, 233)
BLUE = (46, 131, 210)
GREEN = (82, 223, 165)
YELLOW = (251, 201, 91)
RED = (252, 106, 118)


def font(size, bold=False):
    return pygame.font.SysFont('segoeui', size, bold=bold)


def text(surf, value, x, y, size=18, color=WHITE, bold=False):
    surf.blit(font(size, bold).render(str(value), True, color), (int(x), int(y)))


def panel(surf, rect, radius=16):
    pygame.draw.rect(surf, PANEL, rect, border_radius=radius)
    pygame.draw.rect(surf, BORDER, rect, 1, border_radius=radius)


def rounded_button(surf, rect, title, active=False):
    color = (26, 112, 150) if active else (34, 53, 77)
    pygame.draw.rect(surf, color, rect, border_radius=9)
    pygame.draw.rect(surf, CYAN if active else BORDER, rect, 1, border_radius=9)
    label = font(16, True).render(title, True, WHITE)
    surf.blit(label, label.get_rect(center=rect.center))


class Simulator:
    def __init__(self):
        self.dataset_name = 'Dry'
        self.env = None
        self.source = ''
        self.action = 2
        self.auto = False
        self.paused = False
        self.steps_per_second = 2.0
        self.history = deque(maxlen=240)
        self.energy_total = 0.0
        self.reward_total = 0.0
        self.elapsed_steps = 0
        self.particles = []
        self.microbes = []
        self.spawn_clock = 0.0
        self.message = ''
        self.load_env()
        random.seed(7)
        self.microbes = [[random.uniform(285, 915), random.uniform(285, 575), random.uniform(0, 6.28)] for _ in range(17)]

    def load_env(self):
        path = DATASETS[self.dataset_name]
        if path.exists():
            self.env = WastewaterAerationEnv(csv_path=path)
            self.source = path.name
            self.message = 'BSM1-derived influent loaded'
        else:
            self.env = WastewaterAerationEnv()
            self.source = 'SYNTHETIC (dataset not found)'
            self.message = 'No processed CSV found: using synthetic inputs'
        self.reset()

    def reset(self):
        self.env.reset(seed=42)
        self.env.do = 2.5
        self.env.bod = 40.0
        self.env.prev_u = 0.5
        self.env.steps = 0
        self.energy_total = 0.0
        self.reward_total = 0.0
        self.elapsed_steps = 0
        self.history = deque(maxlen=240)
        self.history.append((0, self.env.do, self.env.bod))
        self.particles = []
        self.accumulator = 0.0

    def step(self):
        if self.auto:
            # Deliberately simple feedback baseline, NOT trained DQN/PER-DQN.
            if self.env.do < 2.1:
                self.action = min(4, self.action + 1)
            elif self.env.do > 2.9:
                self.action = max(0, self.action - 1)
        obs, reward, terminated, truncated, info = self.env.step(self.action)
        self.elapsed_steps += 1
        self.energy_total += float(info['energy_proxy'])
        self.reward_total += reward
        self.history.append((self.elapsed_steps * self.env.cfg.dt_hours, self.env.do, self.env.bod))
        if terminated or truncated:
            # Continue a new episode while keeping tank state continuous for visualization.
            # Avoid reset() because it would reset the tank concentrations.
            self.env.steps = 0
        return info

    def animate(self, dt):
        self.spawn_clock += dt
        u = float(self.env.ACTIONS[self.action])
        if self.spawn_clock > 0.035:
            self.spawn_clock = 0.0
            for _ in range(max(1, round(u * 7)) if u > 0 else 0):
                self.particles.append([random.uniform(360, 825), 575.0, random.uniform(1.5, 4.8), random.uniform(65, 160) * (0.5 + u)])
        for p in self.particles:
            p[1] -= p[3] * dt
            p[0] += math.sin(p[1] * 0.03) * 0.25
        self.particles = [p for p in self.particles if p[1] > 275]
        for m in self.microbes:
            m[2] += dt * 1.5


def graph(surf, rect, values, color, ymin, ymax, target=None):
    x, y, w, h = rect
    pygame.draw.rect(surf, (13, 27, 45), rect, border_radius=8)
    for k in range(1, 4):
        yy = y + k * h / 4
        pygame.draw.line(surf, (40, 60, 81), (x, yy), (x + w, yy))
    if target is not None and ymin <= target <= ymax:
        yy = y + h - (target - ymin) / (ymax - ymin) * h
        pygame.draw.line(surf, (135, 146, 164), (x, yy), (x + w, yy), 1)
    if len(values) > 1:
        pts = []
        for i, v in enumerate(values):
            px = x + 6 + i / (len(values) - 1) * (w - 12)
            py = y + h - 5 - max(0, min(1, (v - ymin) / (ymax - ymin))) * (h - 10)
            pts.append((px, py))
        pygame.draw.lines(surf, color, False, pts, 3)


def draw_tank(surf, sim, t):
    # Pipes and arrows
    pygame.draw.rect(surf, (55, 82, 103), (110, 354, 180, 36), border_radius=7)
    pygame.draw.rect(surf, (55, 82, 103), (913, 354, 170, 36), border_radius=7)
    for k in range(3):
        x = 120 + ((t * 55 + k * 52) % 150)
        pygame.draw.polygon(surf, CYAN, [(x, 363), (x + 11, 372), (x, 381)])
        x2 = 935 + ((t * 35 + k * 47) % 135)
        pygame.draw.polygon(surf, GREEN, [(x2, 363), (x2 + 11, 372), (x2, 381)])
    text(surf, 'INFLUENT', 123, 319, 15, MUTED, True)
    text(surf, 'EFFLUENT (tank proxy)', 915, 319, 14, MUTED, True)
    # Tank outline
    pygame.draw.rect(surf, (72, 104, 126), (269, 253, 664, 353), border_radius=15)
    pygame.draw.rect(surf, (23, 89, 135), (281, 271, 640, 321), border_radius=10)
    pygame.draw.rect(surf, (27, 116, 164), (281, 271, 640, 54), border_radius=10)
    for i in range(25):
        x = 286 + i * 26
        yy = 279 + math.sin(t * 2.0 + i * 0.55) * 3
        pygame.draw.circle(surf, (62, 168, 202), (int(x), int(yy)), 2)
    # Organic pollution particles (display count based on concentration)
    rng = random.Random(12)
    count = min(80, max(4, int(sim.env.bod * 0.28)))
    for i in range(count):
        px = rng.randint(295, 900)
        py = rng.randint(306, 560)
        wobble = math.sin(t * 0.65 + i * 0.7) * 4
        pygame.draw.circle(surf, (220, 153, 81), (px, int(py + wobble)), rng.randint(2, 4))
    # Microbes
    for mx, my, phase in sim.microbes:
        yy = my + math.sin(phase) * 5
        pygame.draw.circle(surf, GREEN, (int(mx), int(yy)), 5)
        pygame.draw.circle(surf, (23, 95, 89), (int(mx), int(yy)), 2)
    # Diffuser
    pygame.draw.rect(surf, (20, 48, 68), (341, 576, 520, 14), border_radius=6)
    for i in range(16):
        pygame.draw.circle(surf, CYAN, (355 + 32 * i, 577), 3)
    # Rising air bubbles
    for x, y, r, speed in sim.particles:
        pygame.draw.circle(surf, (161, 230, 247), (int(x), int(y)), int(r), 1)
    # Blower machine
    pygame.draw.rect(surf, (36, 57, 79), (522, 618, 156, 65), border_radius=10)
    pygame.draw.circle(surf, (74, 107, 133), (555, 650), 20)
    for a in range(3):
        angle = t * (2 + sim.action) + a * math.pi * 2 / 3
        pygame.draw.line(surf, CYAN, (555, 650), (555 + 15 * math.cos(angle), 650 + 15 * math.sin(angle)), 5)
    text(surf, 'AIR BLOWER', 580, 636, 16, WHITE, True)
    text(surf, f'{int(sim.env.ACTIONS[sim.action] * 100)}% power', 580, 658, 13, CYAN)
    pygame.draw.line(surf, CYAN, (610, 618), (610, 591), 5)
    text(surf, 'Microorganisms', 291, 292, 13, GREEN)
    text(surf, 'Organic pollution', 738, 292, 13, YELLOW)


def draw(surf, sim, t):
    surf.fill(BG)
    text(surf, 'AERATION LAB', 32, 20, 32, WHITE, True)
    text(surf, 'Interactive wastewater treatment simulator  |  simplified BOD / DO model', 33, 61, 16, MUTED)
    text(surf, 'SOURCE: ' + sim.source, 33, 90, 13, YELLOW if 'SYNTHETIC' in sim.source else GREEN)
    panel(surf, pygame.Rect(24, 124, 1080, 588))
    panel(surf, pygame.Rect(1120, 124, 140, 588))
    text(surf, 'AERATION TANK', 42, 143, 19, WHITE, True)
    text(surf, 'Blower changes oxygen transfer; microbes consume oxygen while removing organic load', 42, 174, 14, MUTED)
    draw_tank(surf, sim, t)
    # Right metrics
    metrics = [
        ('DO', f'{sim.env.do:.2f}', 'mg/L', CYAN),
        ('BOD proxy', f'{sim.env.bod:.1f}', 'mg/L', YELLOW),
        ('Energy proxy', f'{sim.energy_total:.1f}', 'sum u²', GREEN),
        ('Reward', f'{sim.reward_total:.0f}', 'cumulative', WHITE),
        ('Time', f'{sim.elapsed_steps * sim.env.cfg.dt_hours:.1f}', 'hours', WHITE),
    ]
    for i, (label, value, unit, color) in enumerate(metrics):
        yy = 147 + i * 106
        text(surf, label, 1133, yy, 14, MUTED, True)
        text(surf, value, 1133, yy + 24, 26, color, True)
        text(surf, unit, 1133, yy + 58, 13, MUTED)
    # Bottom controls and charts
    panel(surf, pygame.Rect(24, 718, 1236, 54), radius=10)
    text(surf, '0-4 blower   SPACE pause   A auto feedback   R reset   +/- speed   D dry   N rain   S storm', 43, 734, 17, WHITE)
    text(surf, 'DO target: 2–3 mg/L', 44, 202, 15, GREEN)
    text(surf, 'CONTROL: ' + ('AUTO FEEDBACK (not RL)' if sim.auto else 'MANUAL'), 365, 202, 15, CYAN)
    text(surf, 'PAUSED' if sim.paused else f'RUNNING {sim.steps_per_second:g} steps/sec', 720, 202, 15, YELLOW if sim.paused else GREEN)
    # Embedded sparkline graphs
    hist = list(sim.history)
    graph(surf, (42, 622, 300, 64), [r[1] for r in hist], CYAN, 0, 8, target=2.5)
    graph(surf, (365, 622, 300, 64), [r[2] for r in hist], YELLOW, 0, max(150, max((r[2] for r in hist), default=150)))
    text(surf, 'DO HISTORY', 47, 604, 13, MUTED)
    text(surf, 'BOD PROXY HISTORY', 370, 604, 13, MUTED)
    text(surf, sim.message[:52], 692, 639, 13, MUTED)


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('AERATION LAB | Wastewater Aeration Simulator')
    clock = pygame.time.Clock()
    sim = Simulator()
    time_s = 0.0
    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.1)
        time_s += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif pygame.K_0 <= event.key <= pygame.K_4:
                    sim.action = event.key - pygame.K_0
                    sim.auto = False
                elif event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_a:
                    sim.auto = not sim.auto
                elif event.key == pygame.K_r:
                    sim.reset()
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    sim.steps_per_second = min(16, sim.steps_per_second * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim.steps_per_second = max(0.25, sim.steps_per_second / 2)
                elif event.key in (pygame.K_d, pygame.K_n, pygame.K_s):
                    sim.dataset_name = {pygame.K_d: 'Dry', pygame.K_n: 'Rain', pygame.K_s: 'Storm'}[event.key]
                    sim.load_env()
        if not sim.paused:
            sim.accumulator += dt * sim.steps_per_second
            while sim.accumulator >= 1:
                sim.step()
                sim.accumulator -= 1
            sim.animate(dt)
        draw(screen, sim, time_s)
        pygame.display.flip()


if __name__ == '__main__':
    main()
