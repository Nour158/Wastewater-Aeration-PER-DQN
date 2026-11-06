from dataclasses import dataclass

@dataclass
class TankConfig:
    dt_hours: float = 0.25
    volume_m3: float = 1000.0
    k_b_per_hour: float = 0.08
    k_o_mg_l: float = 0.5
    oxygen_per_bod: float = 0.5
    do_saturation_mg_l: float = 8.0
    kla_max_per_hour: float = 2.0
    influent_do_mg_l: float = 0.5
    target_do_mg_l: float = 2.5
    min_do_mg_l: float = 2.0
    bod_limit_mg_l: float = 30.0
    episode_steps: int = 96
    max_do_mg_l: float = 3.0

    low_do_weight: float = 5.0
    high_do_weight: float = 1.0
    bod_weight: float = 2.0
    energy_weight: float = 1.0
    switching_weight: float = 0.1
