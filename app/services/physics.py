"""Post-surrogate physics: wind correction + Lewis drying kinetics + efficiency.

Constants and correlations are sourced from the project dossier (sec 7-8).
Values are engineering approximations adequate for the dryer's operating envelope.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# Air properties at ~310 K, 1 atm
RHO_AIR = 1.13          # kg/m^3
CP_AIR = 1007.0         # J/(kg*K)
MU_AIR = 1.9e-5         # Pa*s
K_AIR = 0.027           # W/(m*K)
PR_AIR = 0.71

# Geometry (dossier sec 3)
COVER_LENGTH = 2.0      # m, characteristic length along wind
COVER_AREA = 1.0        # m^2, ~ 2 m x 0.5 m
INLET_AREA = 0.005      # m^2, conservative inlet cross-section
INLET_VELOCITY = 0.01   # m/s

# Crop kinetics (Lewis), Arrhenius temperature dependence per dossier sec 7.5
R_GAS = 8.314           # J/(mol*K)
CROP_PARAMS = {
    # k0 [1/s], Ea [J/mol]
    "tomato": (0.012, 25_000),
    "mango":  (0.009, 27_000),
    "chilli": (0.015, 24_000),
    "onion":  (0.011, 26_000),
}


@dataclass(frozen=True)
class WindResult:
    delta_t_k: float
    h_wind: float


def wind_correction(t_cover_c: float, ambient_c: float, wind_mps: float) -> WindResult:
    """Flat-plate forced-convection Nu correlation -> external HTC -> tray temp drop.

    Nu = 0.664 * Re^0.5 * Pr^(1/3)   (laminar)
    Returns the temperature drop applied to each tray prediction.
    """
    if wind_mps <= 0:
        return WindResult(delta_t_k=0.0, h_wind=0.0)

    re = RHO_AIR * wind_mps * COVER_LENGTH / MU_AIR
    nu = 0.664 * (re ** 0.5) * (PR_AIR ** (1 / 3))
    h_wind = nu * K_AIR / COVER_LENGTH

    m_dot = RHO_AIR * INLET_VELOCITY * INLET_AREA  # kg/s
    if m_dot <= 0:
        return WindResult(delta_t_k=0.0, h_wind=h_wind)

    delta_t = (h_wind * COVER_AREA * (t_cover_c - ambient_c)) / (m_dot * CP_AIR)
    # Clamp to physically sensible range; correlation extrapolates poorly beyond ~10 K
    delta_t = max(0.0, min(delta_t, 15.0))
    return WindResult(delta_t_k=delta_t, h_wind=h_wind)


def lewis_drying_time(
    tray_temp_c: float,
    crop: str,
    initial_moisture_db: float,
    target_moisture_db: float,
    equilibrium_moisture_db: float = 0.05,
) -> tuple[float, float]:
    """Return (drying_time_hours, k_per_hour) for a single tray using Lewis MR=exp(-kt).

    k(T) = k0 * exp(-Ea / R T)
    """
    k0, ea = CROP_PARAMS.get(crop, CROP_PARAMS["tomato"])
    t_kelvin = tray_temp_c + 273.15
    k_per_s = k0 * math.exp(-ea / (R_GAS * t_kelvin))
    k_per_hour = k_per_s * 3600.0

    mr_target = max(
        (target_moisture_db - equilibrium_moisture_db)
        / (initial_moisture_db - equilibrium_moisture_db),
        1e-6,
    )
    if k_per_hour <= 0:
        return float("inf"), 0.0
    time_h = -math.log(mr_target) / k_per_hour
    return time_h, k_per_hour


def thermal_efficiency(t_outlet_c: float, t_inlet_c: float, heat_flux: float) -> float:
    """eta = m*cp*dT / (I*A_collector). Returns dimensionless fraction."""
    m_dot = RHO_AIR * INLET_VELOCITY * INLET_AREA
    useful_w = m_dot * CP_AIR * max(t_outlet_c - t_inlet_c, 0.0)
    incident_w = heat_flux * COVER_AREA
    if incident_w <= 0:
        return 0.0
    return useful_w / incident_w
