# Using Statistical Regression to Predict Freight Revenue

An exploratory regression analysis using the STB Public Use Carload Waybill Sample to model inflation-adjusted railroad freight revenue.

## Data

**Source:** [Surface Transportation Board (STB) Public Use Waybill Sample](https://www.stb.gov/reports-data/waybill/)

The dataset contains millions of individual rail shipment records with fields covering commodity type, car type, shipment distance, origin/destination geography, car ownership, and reported freight revenue.

**Important caveat:** Roughly two-thirds of waybills have their freight revenue masked by a confidential scalar factor to protect contract rates. This means the reported `Freight Revenue ($)` field is frequently distorted. The models are learning partly from masked values, so R² figures should be interpreted with caution.

## Setup

```r
install.packages(c("data.table", "car"))
library(data.table)
library(car)
```

Place the waybill CSV at `~/Summer 2026 Projects/Public_Use_Carload_Waybill_Sample_20260616.csv` or update the path at the top of the script.

## Key Variables

| Variable | Description |
|---|---|
| `Freight Revenue Real` | Inflation-adjusted revenue (2024 dollars, PPI-deflated) |
| `Number of Carloads` | Cars in the shipment |
| `Estimated Short Line Miles` | Shipment distance |
| `weight_per_car` | Derived: actual tons / number of carloads |
| `Number of Interchanges` | Railroads the shipment crossed |
| `STCC2` | 2-digit commodity code (Standard Transportation Commodity Code) |
| `STB Car Type` | Equipment type code |
| `Car Ownership Code` | P = private, R = railroad, T = TTX/pool |
| `Origin/Termination Freight Rate Territory` | Historical railroad pricing zones |
| `All Rail/Intermodal Code` | All rail, intermodal, or not applicable |

## Data Cleaning

- Dropped ~33 columns with no predictive value (equipment dimensions, interchange states, intermodal unit details, etc.)
- Removed 63 rows missing values in pivotal columns
- Filtered to `Rebill Code == 0` (local/normal shipments only; Rule 11 rebilled shipments removed)
- Removed records with zero freight revenue, zero miles, or zero weight
- Cleaned comma-formatted numeric strings
- Adjusted revenue for inflation using annual PPI averages for railroad freight (base year: 2024)

## Methodology

Revenue follows a heavily right-skewed distribution. A log transformation of the outcome and continuous predictors substantially improved model fit and reduced heteroscedasticity. All models were fit on random samples of 100,000–200,000 rows drawn from the full dataset.

**Geographic encoding:** Freight Rate Territories (coarse, pricing-aligned zones) were used rather than BEA areas. Both were considered; Freight Rate Territories are more directly tied to how railroads historically set rates.

**Multicollinearity:** `Actual Weight in Tons` and `Number of Carloads` were correlated. These were replaced by a single derived feature, `weight_per_car`.

## Models

### Model 1 — Linear baseline (full dataset, 2005–2024)
```
Freight Revenue Real ~ Number of Carloads + Actual Weight in Tons +
  Estimated Short Line Miles + Number of Interchanges + Car Ownership Code +
  STB Car Type + STCC2 + All Rail/Intermodal Code + Hazardous/Bulk Material +
  Origin/Termination Freight Rate Territory + Waybill Month + Type of Move
```
R² = 0.74 | Residual SE = ~$48,930 | Issue: cone-shaped residuals (heteroscedasticity)

---

### Model 2 — Log-linear (full dataset)
Log outcome and log predictors. Waybill Month dropped after backward stepwise selection.

R² = 0.90 | Residual SE = 0.47 (~±59% in dollar terms)

---

### Model 3 — Log-linear with log(weight_per_car) (full dataset)
```
log(Freight Revenue Real) ~ log(Number of Carloads) + log(Estimated Short Line Miles) +
  Number of Interchanges + log(weight_per_car) + Car Ownership Code +
  STB Car Type + STCC2 + All Rail/Intermodal Code + Hazardous/Bulk Material +
  Origin/Termination Freight Rate Territory + Type of Move
```
R² = 0.90 | Residual SE = 0.44 (~±55%) | Q-Q plot shows leptokurtosis (fat tails)

---

### Model 13 — Log-linear, post-2021 data only
Same specification as Model 3, fit only on 2021–2024 data to account for the STB's expanded sampling methodology (effective January 1, 2021) and COVID-era rate shifts.

R² = 0.87 | Residual SE = 0.41 (~±51%)

---

### Model 15 — Short-haul dummy + interaction, no quadratic terms (post-2021)
Short-haul rail pricing is structurally different from long-haul: terminal handling costs are fixed regardless of distance, making per-mile rates much higher for short moves. Model 15 introduces a short-haul dummy variable (< 500 miles) and an interaction between short-haul and number of interchanges, but keeps all continuous predictors first-order. This served as a baseline to test whether adding quadratic terms was warranted.

```
log(Freight Revenue Real) ~ log(Estimated Short Line Miles) + short_haul +
  log(Number of Carloads) + log(weight_per_car) + Number of Interchanges +
  Car Ownership Code + STB Car Type + STCC2 + All Rail/Intermodal Code +
  Hazardous/Bulk Material + Origin/Termination Freight Rate Territory +
  Type of Move + short_haul:Number of Interchanges
```

---

### Model 16 — Log-linear with quadratic terms + short-haul interaction (best model)
Model 16 adds:
- Quadratic terms for log(miles), log(carloads), and log(weight_per_car)

```
log(Freight Revenue Real) ~ log(Estimated Short Line Miles) + short_haul +
  log(Number of Carloads) + log(weight_per_car) + Number of Interchanges +
  Car Ownership Code + STB Car Type + STCC2 + All Rail/Intermodal Code +
  Hazardous/Bulk Material + Origin/Termination Freight Rate Territory +
  Type of Move + I(log(miles)^2) + I(log(carloads)^2) + I(log(weight_per_car)^2) +
  short_haul:Number of Interchanges
```

An ANOVA comparing Model 15 and Model 16 returned a very small p-value, confirming that the quadratic terms add statistically significant explanatory power.

| Metric | Value |
|---|---|
| Train R² | 0.876 |
| Test R² | 0.875 |
| Test RMSE (log scale) | 0.404 |
| Predictions within 15% of actual | 32.9% |

The gap between R² and the 32.9% within-15% figure reflects the fundamental noise introduced by masked contract rates — the model is structurally limited by the data, not by specification.

## Predictions

All dollar predictions use a smearing correction to account for log-normal bias:

```r
exp(predicted_log_value) * exp(0.5 * sigma^2)
```

Example: a single-car, 500-mile hazmat shipment in STCC2 class 28 predicts ~**$6,876** (95% PI: $2,878–$16,429).

## Known Limitations & Future Directions

- **Masked revenues:** ~67% of waybills have factored revenue. This is an irreducible data limitation.
- **Pre/post-2021 structural break:** The STB expanded waybill sampling on January 1, 2021, and COVID caused a large median revenue jump. Separate models for each era are worth exploring.
- **Short-haul vs. long-haul pricing:** Single-car, multi-car, and unit train (100+ cars) shipments likely follow different pricing structures. Splitting data by shipment size could improve accuracy.
- **Commodity granularity:** STCC2 is a 2-digit code. Chemicals (35% of outliers) have highly individualized contract rates that a 2-digit grouping can't capture.
- **Alternative approaches:** Weighted Least Squares, random forests, or gradient boosting may handle the fat-tailed distribution more gracefully than OLS.

## File Structure

```
.
├── README.md
└── freight_revenue_regression.R   # All data cleaning, modeling, and evaluation code
```
