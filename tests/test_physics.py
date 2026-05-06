import math
from app.services.physics import (
    lewis_drying_time,
    thermal_efficiency,
    wind_correction,
)


def test_wind_correction_zero_when_no_wind():
    assert wind_correction(t_cover_c=60, ambient_c=25, wind_mps=0).delta_t_k == 0.0


def test_wind_correction_increases_with_wind_speed():
    a = wind_correction(t_cover_c=60, ambient_c=25, wind_mps=2).delta_t_k
    b = wind_correction(t_cover_c=60, ambient_c=25, wind_mps=6).delta_t_k
    assert b > a >= 0


def test_lewis_drying_time_decreases_with_temperature():
    fast, _ = lewis_drying_time(60, "tomato", 9.0, 0.15)
    slow, _ = lewis_drying_time(35, "tomato", 9.0, 0.15)
    assert slow > fast > 0
    assert math.isfinite(fast)


def test_thermal_efficiency_bounded():
    eff = thermal_efficiency(t_outlet_c=35, t_inlet_c=25, heat_flux=600)
    assert 0.0 <= eff <= 1.0
