from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.session import get_db
from app.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionRecord,
    TrayTemps,
)
from app.services.cache import get_cache
from app.services.physics import wind_correction
from app.services.surrogate import Surrogate, SurrogateInput

router = APIRouter(prefix="/v1", tags=["predict"])


def _predict_temps_k(hf: float, porosity: float) -> tuple[tuple[float, float, float, float], bool]:
    cache = get_cache()
    cached = cache.get(hf, porosity)
    if cached is not None:
        return cached, True
    temps = Surrogate.get().predict(SurrogateInput(hf, porosity))
    cache.set(hf, porosity, temps)
    return temps, False


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    temps_k, was_cached = _predict_temps_k(req.heat_flux, req.porosity)
    temps_c = tuple(t - 273.15 for t in temps_k)

    # Use the hottest tray as a cover-temp proxy for the wind heat-loss calc.
    wind = wind_correction(t_cover_c=temps_c[0], ambient_c=req.ambient_c, wind_mps=req.wind_mps)
    corrected = tuple(t - wind.delta_t_k for t in temps_c)

    crud.record_prediction(
        db,
        heat_flux=req.heat_flux,
        porosity=req.porosity,
        ambient_c=req.ambient_c,
        wind_mps=req.wind_mps,
        temps_c=corrected,
        model_version=settings.model_version,
        cached=was_cached,
    )

    return PredictResponse(
        temps=TrayTemps(t1_c=corrected[0], t2_c=corrected[1], t3_c=corrected[2], t4_c=corrected[3]),
        wind_correction_k=wind.delta_t_k,
        model_version=settings.model_version,
        cached=was_cached,
    )


@router.get("/predictions/recent", response_model=list[PredictionRecord])
def recent(limit: int = 50, db: Session = Depends(get_db)) -> list[PredictionRecord]:
    return [PredictionRecord.model_validate(r) for r in crud.recent_predictions(db, limit=limit)]
