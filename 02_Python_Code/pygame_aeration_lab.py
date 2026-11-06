"""Interactive wastewater aeration lab (Pygame).
Run from project root: python 02_Python_Code/pygame_aeration_lab.py
Requires pygame, numpy, pandas and local wastewater_env.py/config.py.
"""
import csv
import math
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import pygame

from config import TankConfig
from wastewater_env import WastewaterAerationEnv

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / '03_Results' / 'interactive_experiments'
DATA = ROOT / '01_Dataset' / 'processed'
WIDTH, HEIGHT = 1500, 900
FPS = 60
BG = (11, 20, 35)
PANEL = (19, 32, 51)
PANEL2 = (26, 43, 65)
WHITE = (232, 243, 250)
MUTED = (150, 172, 192)
CYAN = (62, 210, 231)
BLUE = (49, 143, 222)
GREEN = (91, 221, 156)
ORANGE = (250, 186, 94)
RED = (252, 105, 113)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('AQUA LAB | Wastewater Aeration Control')
clock = pygame.time.Clock()
F = pygame.font.SysFont('segoeui', 17)
SM = pygame.font.SysFont('segoeui', 14)
BIG = pygame.font.SysFont('segoeui', 30, bold=True)
TITLE = pygame.font.SysFont('segoeui', 24, bold=True)


def text(s, x, y, color=WHITE, font=F):
    screen.blit(font.render(str(s), True, color), (int(x), int(y)))


def card(rect, fill=PANEL, radius=16):
    pygame.draw.rect(screen, fill, rect, border_radius=radius)


def button(rect, label, active=False, disabled=False):
    c = (33, 108, 133) if active else PANEL2
    if disabled:
        c = (33, 43, 55)
    pygame.draw.rect(screen, c, rect, border_radius=9)
    pygame.draw.rect(screen, (56, 92, 117), rect, 1, border_radius=9)
    img = SM.render(label, True, MUTED if disabled else WHITE)
    screen.blit(img, img.get_rect(center=rect.center))


class Slider:
    def __init__(self, label, x, y, lo, hi, value, step, suffix='', format_spec='.2f'):
        self.label, self.x, self.y = label, x, y
        self.lo, self.hi, self.value, self.step = lo, hi, value, step
        self.suffix, self.format_spec = suffix, format_spec
        self.drag = False
        self.w = 318

    def event(self, e):
        r = pygame.Rect(self.x, self.y + 23, self.w, 22)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and r.collidepoint(e.pos):
            self.drag = True
            self.set_x(e.pos[0])
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drag = False
        elif e.type == pygame.MOUSEMOTION and self.drag:
            self.set_x(e.pos[0])

    def set_x(self, mx):
        p = min(1, max(0, (mx - self.x) / self.w))
        self.value = round((self.lo + p * (self.hi - self.lo)) / self.step) * self.step
        self.value = min(self.hi, max(self.lo, self.value))

    def draw(self):
        text(self.label, self.x, self.y - 2, WHITE, SM)
        val = format(self.value, self.format_spec) + self.suffix
        img = SM.render(val, True, CYAN)
        screen.blit(img, (self.x + self.w - img.get_width(), self.y - 2))
        yy = self.y + 34
        pygame.draw.line(screen, (65, 89, 111), (self.x, yy), (self.x + self.w, yy), 6)
        xx = self.x + int(self.w * (self.value - self.lo) / (self.hi - self.lo))
        pygame.draw.line(screen, CYAN, (self.x, yy), (xx, yy), 6)
        pygame.draw.circle(screen, WHITE, (xx, yy), 9)


sliders = {
    'blower': Slider('Blower power (manual)', 1110, 154, 0, 100, 50, 25, '%', '.0f'),
    'target': Slider('Target DO', 1110, 217, 1, 5, 2.5, .1, ' mg/L', '.1f'),
    'kb': Slider('Biodegradation rate k_b', 1110, 280, .01, .25, .08, .01, ' /h'),
    'kla': Slider('Maximum oxygen transfer KLa', 1110, 343, .5, 5, 2, .1, ' /h', '.1f'),
    'volume': Slider('Tank volume', 1110, 406, 500, 5000, 1000, 100, ' m3', '.0f'),
    'energy': Slider('Energy reward weight', 1110, 469, 0, 8, 1, .25, '', '.2f'),
    'high_do': Slider('High DO reward weight', 1110, 532, 0, 8, 1, .25, '', '.2f'),
}

MODE_RECTS = {
    'Manual': pygame.Rect(1110, 625, 100, 34),
    'Auto DO': pygame.Rect(1218, 625, 100, 34),
    'PER-DQN': pygame.Rect(1326, 625, 100, 34),
}
DATA_RECTS = {
    'dry': pygame.Rect(1109, 698, 94, 33),
    'rain': pygame.Rect(1213, 698, 94, 33),
    'storm': pygame.Rect(1317, 698, 108, 33),
}
RUN_RECT = pygame.Rect(1110, 757, 96, 40)
RESET_RECT = pygame.Rect(1214, 757, 96, 40)
SAVE_RECT = pygame.Rect(1318, 757, 108, 40)


def data_path(kind):
    candidates = {
        'dry': ['dry_influent.csv', 'Inf_dry_2006.csv'],
        'rain': ['rain_influent.csv', 'Inf_rain_2006.csv'],
        'storm': ['storm_influent.csv', 'Inf_strm_2006.csv'],
    }
    for filename in candidates[kind]:
        p = DATA / filename
        if p.exists():
            return p
    return None


kind = 'dry'
mode = 'Manual'
paused = False
speed = 4  # simulated steps per second
accum = 0
message = 'Ready. Adjust sliders or choose Auto DO.'
rows = []
history = deque(maxlen=250)


def make_env():
    cfg = TankConfig()
    p = data_path(kind)
    return WastewaterAerationEnv(csv_path=p, config=cfg), ('BSM1-derived ' + kind if p else 'Synthetic fallback')


def reset():
    global env, source, obs, rows, history, accum, message
    env, source = make_env()
    obs, _ = env.reset(seed=42)
    # Ensure the chosen controls are applied at reset, too.
    apply_config()
    rows = []
    history = deque(maxlen=250)
    history.append((0, env.do, env.bod, 0, 0))
    accum = 0
    message = 'Reset: ' + source + '. Same seed for repeatability.'


def apply_config():
    cfg = env.cfg
    cfg.k_b_per_hour = sliders['kb'].value
    cfg.kla_max_per_hour = sliders['kla'].value
    cfg.volume_m3 = sliders['volume'].value
    cfg.target_do_mg_l = sliders['target'].value
    cfg.energy_weight = sliders['energy'].value
    cfg.high_do_weight = sliders['high_do'].value
    # Upper bound follows target + 0.5, lower bound remains 2 mg/L.
    cfg.max_do_mg_l = max(cfg.min_do_mg_l + .1, cfg.target_do_mg_l + .5)


def action_index():
    if mode == 'Manual':
        return int(round(sliders['blower'].value / 25))
    if mode == 'Auto DO':
        # Simple feedback baseline, NOT trained AI.
        error = sliders['target'].value - env.do
        desired = min(1., max(0., env.prev_u + .3 * error))
        return int(min(4, max(0, round(desired * 4))))
    return 2


def advance():
    global obs, message, paused
    apply_config()
    a = action_index()
    obs, reward, terminated, truncated, info = env.step(a)
    t = env.steps * env.cfg.dt_hours
    history.append((t, env.do, env.bod, info['blower_fraction'], reward))
    row = {
        'time_hours': round(t, 4), 'mode': mode, 'dataset': source,
        'DO_mg_L': env.do, 'BOD_proxy_mg_L': env.bod,
        'blower_fraction': info['blower_fraction'],
        'influent_BOD_proxy_mg_L': info['influent_bod_proxy_mg_l'],
        'flow_m3_h': info['flow_m3_h'],
        'energy_proxy': info['energy_proxy'], 'reward': reward,
        'k_b_per_hour': env.cfg.k_b_per_hour,
        'KLa_max_per_hour': env.cfg.kla_max_per_hour,
        'volume_m3': env.cfg.volume_m3,
        'target_DO_mg_L': env.cfg.target_do_mg_l,
        'energy_weight': env.cfg.energy_weight,
        'high_DO_weight': env.cfg.high_do_weight,
    }
    rows.append(row)
    if truncated or terminated:
        paused = True
        message = '24-hour episode complete. Save CSV or press Reset.'


def save_csv():
    global message
    if not rows:
        message = 'Nothing to save yet. Run a few simulation steps.'
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / ('experiment_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    message = 'Saved CSV: ' + str(out)
    print(message)


def plot(rect, key, color, lo=None, hi=None, target=None):
    pygame.draw.rect(screen, PANEL2, rect, border_radius=10)
    values = [p[key] for p in history]
    if len(values) < 2:
        return
    mn = min(values) if lo is None else lo
    mx = max(values) if hi is None else hi
    if mx - mn < .01:
        mx = mn + 1
    if target is not None and mn <= target <= mx:
        ty = rect.bottom - 12 - int((target - mn) / (mx - mn) * (rect.height - 24))
        pygame.draw.line(screen, (100, 120, 135), (rect.x + 8, ty), (rect.right - 8, ty), 1)
    pts = []
    for i, v in enumerate(values):
        xx = rect.x + 9 + i * (rect.width - 18) / (len(values) - 1)
        yy = rect.bottom - 12 - (v - mn) / (mx - mn) * (rect.height - 24)
        pts.append((int(xx), int(yy)))
    pygame.draw.lines(screen, color, False, pts, 2)
    text(f'{mn:.1f}', rect.x + 8, rect.bottom - 21, MUTED, SM)
    text(f'{mx:.1f}', rect.x + 8, rect.y + 4, MUTED, SM)


particles = [(random.random(), random.random(), random.random()) for _ in range(85)]


def draw_tank(t):
    r = pygame.Rect(45, 156, 635, 384)
    card(r, PANEL)
    text('AERATION BASIN  |  LIVE PROCESS', 65, 173, WHITE, TITLE)
    # Pipes
    pygame.draw.rect(screen, (104, 130, 147), (54, 262, 100, 30), border_radius=6)
    pygame.draw.rect(screen, (104, 130, 147), (577, 280, 88, 30), border_radius=6)
    text('INLET', 64, 241, MUTED, SM)
    text('OUTLET', 587, 259, MUTED, SM)
    tank = pygame.Rect(137, 229, 462, 269)
    pygame.draw.rect(screen, (97, 124, 147), tank, border_radius=14)
    water = tank.inflate(-12, -12)
    pygame.draw.rect(screen, (25, 91, 135), water, border_radius=9)
    # Water bands and wave line
    for i in range(6):
        y = water.y + 25 + i * 34
        pygame.draw.line(screen, (29 + i * 2, 103 + i * 3, 148 + i * 2), (water.x, y), (water.right, y), 1)
    u = env.prev_u
    # Aeration diffusers
    pygame.draw.line(screen, (181, 204, 213), (water.x + 40, water.bottom - 20), (water.right - 40, water.bottom - 20), 7)
    for i in range(7):
        x = water.x + 58 + i * 55
        pygame.draw.circle(screen, (174, 198, 211), (x, water.bottom - 20), 7)
    # Bubbles scale with aeration
    count = int(90 * u)
    for i in range(count):
        px, py, phase = particles[i % len(particles)]
        xx = water.x + 38 + px * (water.width - 76)
        yy = water.bottom - 28 - ((py * 190 + t * (42 + 75 * u) + phase * 120) % 204)
        rad = 2 + int(phase * 4)
        pygame.draw.circle(screen, (148, 229, 246), (int(xx), int(yy)), rad, 1)
    # Pollutant and microbes (illustrative visual only)
    pollutant_count = min(55, int(10 + env.bod / 3))
    for i in range(pollutant_count):
        px, py, phase = particles[(i + 10) % len(particles)]
        xx = water.x + 14 + px * (water.width - 28)
        yy = water.y + 20 + py * (water.height - 50)
        pygame.draw.circle(screen, (234, 163, 93), (int(xx), int(yy)), 2)
    for i in range(18):
        px, py, phase = particles[(i + 60) % len(particles)]
        xx = water.x + 18 + px * (water.width - 36)
        yy = water.y + 24 + py * (water.height - 54)
        pygame.draw.circle(screen, GREEN, (int(xx), int(yy)), 4, 1)
    text('Orange = organic load    Green = microbes    Blue = oxygen bubbles', 64, 510, MUTED, SM)


def draw():
    screen.fill(BG)
    text('WASTEWATER AERATION  /  INTERACTIVE LAB', 40, 30, WHITE, BIG)
    text('Simplified BOD/DO process model  •  BSM1-derived influent when available', 42, 72, MUTED, SM)
    card(pygame.Rect(36, 111, 1038, 735))
    card(pygame.Rect(1090, 111, 360, 735))
    text('LIVE CONTROL PANEL', 1110, 125, WHITE, TITLE)
    t = pygame.time.get_ticks() / 1000
    draw_tank(t)
    # Metrics
    for x, label, val, c in [
        (708, 'DISSOLVED OXYGEN', f'{env.do:.2f} mg/L', CYAN),
        (708, 'BOD PROXY', f'{env.bod:.2f} mg/L', ORANGE),
        (708, 'BLOWER', f'{env.prev_u * 100:.0f}%', GREEN),
        (708, 'TIME', f'{env.steps * env.cfg.dt_hours:.2f} h', WHITE),
    ]:
        y = 160 + ('DISSOLVED OXYGEN', 'BOD PROXY', 'BLOWER', 'TIME').index(label) * 95
        card(pygame.Rect(x, y, 330, 81), PANEL2, 10)
        text(label, x + 17, y + 12, MUTED, SM)
        text(val, x + 17, y + 36, c, TITLE)
    text('DO history', 54, 557, WHITE, F)
    plot(pygame.Rect(50, 585, 480, 105), 1, CYAN, 0, 8, sliders['target'].value)
    text('BOD proxy history', 558, 557, WHITE, F)
    plot(pygame.Rect(552, 585, 480, 105), 2, ORANGE)
    text('Energy proxy (sum of u²)', 54, 715, MUTED, SM)
    text(f'{sum(r["energy_proxy"] for r in rows):.2f}', 54, 740, GREEN, TITLE)
    text('Cumulative reward', 316, 715, MUTED, SM)
    text(f'{sum(r["reward"] for r in rows):.2f}', 316, 740, ORANGE, TITLE)
    text('Influent BOD proxy', 575, 715, MUTED, SM)
    text(f'{float(obs[3]):.1f} mg/L', 575, 740, WHITE, TITLE)
    text('Influent flow', 805, 715, MUTED, SM)
    text(f'{float(obs[4]):.1f} m³/h', 805, 740, WHITE, TITLE)
    for s in sliders.values():
        s.draw()
    text('CONTROL MODE', 1110, 598, MUTED, SM)
    for name, r in MODE_RECTS.items():
        button(r, name, active=(name == mode), disabled=(name == 'PER-DQN'))
    text('INFLUENT DATASET', 1110, 675, MUTED, SM)
    for name, r in DATA_RECTS.items():
        button(r, name.upper(), active=(name == kind))
    button(RUN_RECT, 'Resume' if paused else 'Pause', active=not paused)
    button(RESET_RECT, 'Reset')
    button(SAVE_RECT, 'Save CSV')
    text(f'Speed: {speed} steps/sec  |  +/- to adjust', 1110, 808, MUTED, SM)
    text(message[:195], 41, 863, MUTED, SM)
    pygame.display.flip()


reset()
running = True
while running:
    dt = clock.tick(FPS) / 1000
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_SPACE:
                paused = not paused
            elif e.key == pygame.K_r:
                reset()
            elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                speed = min(32, speed * 2)
            elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                speed = max(1, speed // 2)
            elif e.key in (pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                sliders['blower'].value = (e.key - pygame.K_0) * 25
                mode = 'Manual'
        for s in sliders.values():
            s.event(e)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if RUN_RECT.collidepoint(e.pos):
                paused = not paused
            elif RESET_RECT.collidepoint(e.pos):
                reset()
            elif SAVE_RECT.collidepoint(e.pos):
                save_csv()
            for name, r in MODE_RECTS.items():
                if r.collidepoint(e.pos):
                    if name == 'PER-DQN':
                        message = 'PER-DQN is not trained yet. Manual and Auto DO are available.'
                    else:
                        mode = name
                        message = name + ' control enabled.'
            for name, r in DATA_RECTS.items():
                if r.collidepoint(e.pos) and name != kind:
                    kind = name
                    reset()
    if not paused:
        accum += dt * speed
        while accum >= 1 and not paused:
            advance()
            accum -= 1
    draw()
pygame.quit()
sys.exit(0)
