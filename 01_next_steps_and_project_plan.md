# Solar Dryer Digital Twin — Next Steps & Full Project Plan
*CLL251: Heat Transfer for Chemical Engineers — IIT Delhi*
*Status doc, generated 06 May 2026*

---

## 1. Project at a Glance

**One-liner.** A CFD-trained surrogate model for real-time thermal prediction of an agricultural solar dryer, deployed as an interactive web app. Users adjust solar irradiation, ambient conditions, and crop properties via sliders and instantly see predicted tray temperatures, drying time, and thermal efficiency — without running CFD themselves.

**Why this framing works for grading.** The submitted abstract committed to a numerical model of transient heat transfer using FDM, with conduction, convection, and solar radiation. Every word of that is honored. The expansion is additive: CFD adds fidelity, ML enables real-time inference, the web app makes it usable. The professor of a heat transfer course sees heat transfer content (derivations, dimensionless analysis, FDM solver) and *also* gets a clickable demo, which is rare and memorable.

**Three layers of modeling, each independently defensible:**
1. Analytical heat balance (closed-form steady-state, Nu correlations)
2. 1D FDM Python solver (transient, explicit Euler)
3. CFD parametric sweep (Ansys Fluent, 102 design points)
4. ML surrogate (polynomial regression / gradient boosting, trained on CFD)

The poster centerpiece is a parity plot showing all four agreeing within a defensible error band.

---

## 2. Where We Are Now (Audit)

### ✅ Completed

**CFD parametric sweep — 102 design points.**
- Heat flux: 300–800 W/m² (28 unique values)
- Crop porosity: 0.5–0.9 (9 unique values)
- Outputs: 4 area-weighted average tray temperatures (K)
- Stored in `solar_dryer_ml.csv`
- Data is physically consistent: T₁ > T₂ > T₃ > T₄ holds for all 102 rows (bottom tray hottest, top tray coolest, as expected for a chimney dryer)

**CFD setup documented.**
- Geometry: slanted absorber + vertical chimney with 4 horizontal trays. Glass cover ≈ 2 m × 0.5 m. Tray 1 is bottom-most, Tray 4 is top-most.
- Mesh: 603,822 elements (single mesh, see Gap below).
- Inlet BC: velocity inlet, magnitude 0.01 m/s normal to boundary (induced flow regime).
- Solver: viscous laminar model (acknowledged compromise for time constraints — flagged on poster).
- Trays modeled as porous media to represent crop bed; porosity is a sweep variable.
- Heat flux applied to absorber plate as boundary condition.

**Visual outputs available.**
- Geometry render
- Mesh render (603,822 elements)
- Static temperature contour at low heat flux, ~300 W/m², porosity 0.8 (peak ~358 K)
- Static temperature contour at high heat flux (peak ~395 K) — the contrast figure
- Velocity vector field through chimney (peak ~16.5 mm/s)
- Tabular tray-temperature output (area-weighted averages)

### ⚠️ Gap: Mesh Independence

We have one mesh at 603,822 elements but no second mesh for comparison. This is the only outstanding CFD task. Run **one** additional case at a representative parameter combo (suggested: 500 W/m², porosity 0.8) on a coarser mesh — target ~250–350k elements — and report the percent change in Tray-1 temperature. A delta below ~1% lets us state on the poster: *"Tray temperatures changed by <X% between coarse and fine meshes; results are mesh-independent within reported precision."* Time cost: one Fluent run + one number on a slide.

### ❌ Not Yet Started

| Track | What's missing |
|---|---|
| ML surrogate | Training, hyperparameter selection, parity plot, error analysis, serialized model |
| FDM module | 1D unsteady absorber-plate solver, validation against CFD steady-state |
| Heat transfer derivations | Energy balances, Ra/Re/Nu analysis, drying kinetics module |
| Web app | Streamlit UI, integration of surrogate + kinetics + wind correction, deployment |
| Poster | Layout, figure integration, copy |
| Presentation | Slide deck, demo flow, rehearsal |

---

## 3. Critical Path & Sequencing

Dependencies determine the order. The ML surrogate is the bottleneck for the web app and for the poster centerpiece (parity plot), so it goes first. FDM and heat transfer derivations can run in parallel — they don't block anything but each other (and they're independent of each other).

```
  ┌─────────────────────────┐
  │ 1. ML Surrogate         │ ← unblocks web app + poster
  └────────────┬────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
 ┌──────────┐    ┌──────────────┐
 │ 4. Web   │    │ 2. FDM       │  ← parallel
 │   App    │    │ 3. HT derivs │  ← parallel
 └─────┬────┘    └──────┬───────┘
       │                │
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │ 5. Poster +    │
       │    Slides      │
       └────────────────┘
```

Mesh-independence run can happen at any time and is non-blocking.

---

## 4. Detailed Task Breakdowns

### Task 1 — ML Surrogate (Priority: blocking)

**Inputs.** `solar_dryer_ml.csv` (102 rows, 2 features, 4 targets).

**Train/test split.** Hold out ~20 points (≈20%) as test set. Use random split with fixed seed for reproducibility. Avoid holding out rows at the *boundary* of the feature space (300 W/m² or 800 W/m²) — train should span the full input range.

**Models to try, in order:**
1. **Polynomial regression, degree 2.** Closed-form, deterministic, almost certainly sufficient given smooth physics over a 2D input space. Baseline.
2. **Polynomial regression, degree 3.** If degree 2 leaves residual structure.
3. **Gradient boosting (sklearn `HistGradientBoostingRegressor` or XGBoost).** Backup if polynomial underfits.
4. **Small MLP (2-input → 16 → 16 → 4).** Only if everything above underperforms. Unlikely to be needed; harder to defend in viva than a polynomial.

**Approach.** Train one model per tray (4 separate regressors) OR a multi-output model — both work. Multi-output is cleaner; per-tray gives slight flexibility.

**Acceptance criteria.**
- Test R² > 0.99 on each tray temperature
- Mean absolute error < 0.5 K on held-out CFD points
- No systematic bias visible on parity plot

**Deliverables.**
- Jupyter notebook with full training pipeline (`surrogate_training.ipynb`)
- Serialized model file (joblib or pickle, `surrogate.pkl`)
- Parity plot: predicted vs CFD truth, all 4 trays on one figure (poster centerpiece)
- Residual histogram and a heatmap of error over the (heat flux × porosity) plane
- A `predict(heat_flux, porosity) → (T1, T2, T3, T4)` function callable in milliseconds

**Time estimate.** 3–5 hours including plots and write-up.

---

### Task 2 — 1D FDM Python Module

**Physics.** Lumped 1D unsteady energy balance on the absorber plate:

$$
\rho \cdot c_p \cdot \delta \cdot \frac{\partial T}{\partial t} = \alpha \cdot I - h \cdot (T - T_\infty) - \varepsilon \cdot \sigma \cdot (T^4 - T_{sky}^4)
$$

Where:
- $\rho$, $c_p$, $\delta$ = density, specific heat, thickness of absorber plate
- $\alpha$ = solar absorptivity
- $I$ = solar irradiance (input)
- $h$ = convective heat transfer coefficient (from Nu correlation)
- $\varepsilon$ = emissivity
- $T_{sky}$ = sky temperature, ≈ $T_\infty - 6$ K rule of thumb

**Method.** Explicit Euler in NumPy. Time step from CFL-like stability bound: $\Delta t < \frac{\rho c_p \delta}{h}$ with safety factor 0.5.

**Validation.** Run to steady state at $I = 300$ W/m², compare plate temp to the CFD case at the same heat flux. Then repeat at $I = 800$ W/m². Report agreement percentage.

**Deliverables.**
- `fdm_absorber.py` script
- Plot: plate temperature vs time, multiple $I$ values overlaid
- Comparison table: FDM steady-state vs CFD vs analytical, three irradiance levels
- Brief markdown writeup (½ page) of derivation, discretization, stability check

**Time estimate.** 4–6 hours.

---

### Task 3 — Heat Transfer Derivations Document

**Sections, in order:**

1. **Energy balance on absorber plate.** Differential and lumped forms. Justification of lumped assumption (Biot < 0.1).

2. **Lumped balance on each tray.** Air-side convective gain from rising plume, conductive loss to walls, latent heat sink from moisture evaporation. Justify why steady CFD captures this adequately.

3. **Coupled mass-energy balance for the crop bed.** Moisture flux out = evaporation rate driven by surface temperature and air humidity. Treat crop layer as porous medium with effective thermal conductivity.

4. **Dimensionless analysis in the chimney.**
   - Rayleigh number (buoyancy-driven flow): $Ra = \frac{g \beta \Delta T L^3}{\nu \alpha_{th}}$
   - Reynolds number based on inlet velocity 0.01 m/s
   - Conclude regime: forced flow at very low Re → likely transitional/laminar mixed convection. Cross-reference with the laminar viscous model used in CFD.

5. **Nusselt correlations.**
   - Flat plate forced convection: $Nu = 0.664 \cdot Re^{0.5} \cdot Pr^{1/3}$ (laminar, used for wind correction in web app)
   - Vertical channel free convection: $Nu = 0.59 \cdot Ra^{0.25}$ (laminar)
   - Combined mixed convection: $Nu_{eff}^3 = Nu_{forced}^3 + Nu_{free}^3$ for assisting flow

6. **Drying kinetics — Lewis thin-layer model.**
   - Moisture ratio: $MR = \frac{M - M_e}{M_0 - M_e} = \exp(-k \cdot t)$
   - Temperature dependence (Arrhenius): $k = k_0 \cdot \exp(-E_a / R T)$
   - Crop-specific constants for default crop (tomato slices): $k_0$, $E_a$ from literature
   - Bridge: tray temperatures from surrogate → $k(T)$ → drying time to target $MR$

**Deliverables.**
- `heat_transfer_derivations.md` (or .pdf rendered from LaTeX)
- 3–5 pages, equation-heavy, with citations to standard textbooks (Incropera, Bergman) and crop-drying literature for kinetic constants

**Time estimate.** 5–8 hours, mostly writing.

---

### Task 4 — Streamlit Web App

**Stack.** Streamlit + scikit-learn (for surrogate) + NumPy + Matplotlib/Plotly. Deploy to Streamlit Community Cloud, free tier, ~10 min from a public GitHub repo.

**Inputs (sidebar sliders/widgets):**
| Input | Range | Default | Notes |
|---|---|---|---|
| Solar irradiation $I$ | 300–800 W/m² | 600 | Direct surrogate input |
| Ambient temperature $T_\infty$ | 15–40 °C | 25 | Used for Δ in efficiency calc and FDM |
| Wind speed $v$ | 0–8 m/s | 2 | Analytical correction (see below) |
| Crop porosity | 0.5–0.9 | 0.7 | Direct surrogate input |
| Crop type | dropdown | Tomato | Selects $k_0$, $E_a$, $M_0$, $M_e$ |
| Target moisture | 5–20% | 10 | Sets endpoint of drying curve |

**Outputs (main panel):**
- 4 large numbers: $T_1$, $T_2$, $T_3$, $T_4$ predicted (°C)
- Drying time to target moisture (hours), per tray
- Thermal efficiency: $\eta = \dot{m} c_p \Delta T / (I \cdot A)$
- 2D color-coded cross-section showing tray temperatures (illustrative, not from CFD)
- Live moisture-vs-time curve, one line per tray
- Small panel: "Wind correction applied: ΔT_wind = X K"

**Wind speed handling.** The CFD dataset doesn't include wind speed as a variable. Model its effect via the flat-plate forced-convection Nu correlation: $Nu = 0.664 \cdot Re^{0.5} \cdot Pr^{1/3}$, which adjusts $h$ on the dryer's external glass cover. The corrected heat-loss term lowers each tray temperature by an analytically computed offset. **State this explicitly on the Discussion panel** of the poster: it's a physics-grounded correction, not a hack.

**Architecture sketch.**
```
ui_inputs ──► surrogate.predict() ──► T1..T4 (CFD-trained)
                                          │
ui_inputs ──► wind_correction() ──── ΔT_wind
                                          │
                                          ▼
                                    T1..T4_corrected
                                          │
                                          ▼
                                    lewis_kinetics(T) ──► drying time
                                          │
                                          ▼
                                    plots + metrics
```

**Deliverables.**
- `app.py` (single-file Streamlit script)
- `requirements.txt`
- `surrogate.pkl`
- Public URL on Streamlit Cloud
- README with deployment steps

**Time estimate.** 6–10 hours including UI polish.

---

### Task 5 — Poster

**Format.** 44" × 44" template provided by course.

**Suggested panel layout (10 panels):**
1. Title + team + course
2. Abstract / one-liner (matches submitted proposal language)
3. Geometry + photo of physical reference (if any)
4. Problem framing: why solar drying, why thermal modeling
5. Methods: CFD setup (mesh, BCs, solver) + governing equations
6. Methods: FDM derivation + ML surrogate approach
7. Results: contour comparison (low vs high heat flux)
8. Results: parity plot (CFD vs ML, the centerpiece)
9. Results: web app screenshot + drying-time prediction examples
10. Discussion (assumptions, wind correction note, limitations) + Conclusion + References + QR code to web app

**Time estimate.** 6–10 hours, mostly figure refinement and copy.

---

### Task 6 — Presentation

- 10–12 slides
- 2 slides for problem and motivation
- 2 slides for CFD methods + results
- 2 slides for FDM + analytical
- 2 slides for ML approach + parity plot
- 2 slides for live web-app demo (screenshare with sliders being moved)
- 1 slide for limitations and future work
- 1 slide closing + Q&A

Rehearse the demo specifically — that's the moment that makes the project memorable. Have a fallback screenshot in case Wi-Fi flakes.

**Time estimate.** 4–6 hours.

---

## 5. Workstream Allocation

Four parallel tracks. Owners TBD; some tracks can be combined if bandwidth is low.

| Track | Owner | Tasks | Dependencies |
|---|---|---|---|
| **A. CFD validation** | TBD | Mesh-independence run, finalize contour figures | None |
| **B. Heat transfer + FDM** | TBD | Tasks 2 + 3 | None |
| **C. Surrogate + Web app** | TBD | Tasks 1 + 4 | Surrogate before web app |
| **D. Deliverables** | TBD | Tasks 5 + 6 | All other tracks |

**Recommended team distribution if 4 people:** one per track. If 2 people: A+D as one track (light), B as second track (medium), C as third track (heavy) — needs the most owner attention.

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Mesh independence shows >5% delta | Low | Medium | Re-run at finer mesh; if persistent, report honestly with disclaimer |
| Polynomial surrogate underfits | Low | Low | Escalate to gradient boosting; data is smooth, very unlikely to need MLP |
| Streamlit Cloud quota issues | Low | Low | Have local-host backup; print QR code linking to GitHub repo as fallback |
| FDM doesn't match CFD within 5% | Medium | Low | Acknowledge as expected — 1D vs 3D will differ; report the gap as a learning |
| Drying-kinetics constants for chosen crop are uncertain | Medium | Low | Cite source clearly; allow crop selection in app to demonstrate robustness |
| Time crunch near deadline | High | High | Lock down poster figures by Day –3; demo rehearsal Day –1 |

---

## 7. Definition of Done

- [ ] Streamlit app live at a public URL, with all 6 input controls functional
- [ ] Parity plot generated, R² > 0.99 on test set, included on poster
- [ ] Mesh-independence statement on poster with concrete number
- [ ] FDM Python script in repo, validated against ≥1 CFD case
- [ ] Heat transfer derivations document in repo, ≥3 pages with equations
- [ ] Poster filled into provided 44×44 template, ready to print
- [ ] Slide deck rehearsed, demo flow practiced ≥1 time
- [ ] GitHub repo cleaned: README with deployment steps, no stale notebooks, dependencies pinned
- [ ] All four physics layers (analytical, FDM, CFD, ML) are present and cross-validated

---

## 8. Communication & Handoff

**Repo.** Single GitHub repo, shared branch model: `main` for stable, feature branches for in-progress work, PR review optional given small team.

**File conventions.**
- All notebooks have a top markdown cell explaining purpose, inputs, outputs
- All scripts have docstrings on every function
- Plots saved as both PNG (for poster) and SVG (for vector embedding)
- Data in `data/`, code in `src/`, figures in `figures/`, docs in `docs/`

**Synchronization.** Daily 15-min standup until deadline week, then twice daily. WhatsApp for ad-hoc.

**Context handoff to AI agents.** Use `02_technical_context_dossier.md` as the single source of truth for any new agent or team member spinning up. It contains the complete technical state of the project.

---

## 9. Immediate Next 24 Hours

If only one thing happens in the next day, it's **train the surrogate.** It unblocks the web app and produces the parity plot. Concretely:
1. Open `solar_dryer_ml.csv`
2. Train/test split (80/20, fixed seed)
3. Fit polynomial regression deg 2 and deg 3
4. Pick winner on test R²
5. Generate parity plot
6. Serialize model

Tasks 2 (FDM) and 3 (derivations) can be picked up in parallel by other team members without blocking each other.
