# =============================================================================
# Using Statistical Regression to Predict Freight Revenue
# Data: STB Public Use Carload Waybill Sample
# =============================================================================

library(data.table)

# Load data
dt <- fread("~/Summer 2026 Projects/Public_Use_Carload_Waybill_Sample_20260616.csv")

# NOTE: Roughly two-thirds of waybills have freight revenue masked via a
# confidential scalar factor to protect contract rates. Reported Freight
# Revenue ($) is often overstated or distorted. The model learns partly from
# masked/factored numbers. Be cautious of R^2 interpretations.


# =============================================================================
# 1. DROP UNIMPORTANT COLUMNS
# =============================================================================

cols_to_drop <- c(
  "Transit Charges ($)",
  "Miscellaneous Charges",
  "Stratum Identification",
  "Subsample Code Number",
  "Car Capacity",
  "Tare Weight of Car",
  "Outside Length",
  "Outside Width",
  "Outside Height",
  "Extreme Outside Height",
  "Type of Wheel Bearings and Brakes",
  "Number of Axles",
  "Draft Gear",
  "Number of Articulated Units",
  "Intermodal (TOFC/COFC) Service Code",
  "Number of TOFC/COFC Units",
  "Intermodal Unit Ownership Code",
  "Intermodal Unit Type Code",
  "Shipment Size Category",
  "AAR Error Codes",
  "Routing Error Flag",
  "Interchange State #1",
  "Interchange State #2",
  "Interchange State #3",
  "Interchange State #4",
  "Interchange State #5",
  "Interchange State #6",
  "Interchange State #7",
  "Interchange State #8",
  "Interchange State #9",
  "Accounting Period",
  "AAR Mechanical Designation",
  "Billed Weight in Tons"
)

dt <- dt[, !cols_to_drop, with = FALSE]


# =============================================================================
# 2. REMOVE ROWS MISSING IMPORTANT FIELDS
# =============================================================================

pivotal_cols <- c(
  "Freight Revenue ($)",
  "Actual Weight in Tons",
  "Estimated Short Line Miles",
  "STCC2",
  "STCC5",
  "STB Car Type",
  "Origin Freight Rate Territory",
  "TerminationFreightRateTerritory"
)

remove_incomplete_rows <- function(dt, cols) {
  keep <- rep(TRUE, nrow(dt))

  for (col in cols) {
    if (is.numeric(dt[[col]]) || is.integer(dt[[col]])) {
      keep <- keep & !is.na(dt[[col]])
    } else {
      keep <- keep & (dt[[col]] != "")
    }
  }

  cat("Rows removed:", sum(!keep), "\n")
  cat("Rows remaining:", sum(keep), "\n")
  return(dt[keep])
}

dt <- remove_incomplete_rows(dt, pivotal_cols)
# Result: 63 rows removed, 18,730,547 remaining


# =============================================================================
# 3. FILTER TO NON-REBILLED SHIPMENTS
# =============================================================================

# Rebill Code values:
#   0 = Local shipment or normal through-rate (not a Rule 11 shipment)
#   1 = Originated - Delivered Rule 11 shipment
#   2 = Received - Delivered Rule 11 shipment
#   3 = Received - Terminated Rule 11 shipment

# See distribution of rebill codes
table(dt$`Rebill Code`)

# Keep only local/normal shipments (rebill code == 0)
dt1 <- dt[dt$`Rebill Code` == 0]


# =============================================================================
# 4. EXPLORE AND ENCODE INTERMODAL CODE
# =============================================================================

# Check whether unreported intermodal codes were an old reporting practice
table(dt1$`Data Year`, dt1$`All Rail/Intermodal Code`)

# Treat as a three-level factor (not a historical artifact)
dt1$`All Rail/Intermodal Code` <- factor(dt1$`All Rail/Intermodal Code`,
  levels = c(1, 2, 9),
  labels = c("All Rail", "Intermodal", "Not Applicable"))

nrow(dt1)  # ~12 million rows


# =============================================================================
# 5. DROP LOW-VALUE COLUMNS AND CLEAN NUMERIC FIELDS
# =============================================================================

# Drop infrequently meaningful columns
dt1[, `Type of Move Via Water (inferred)` := NULL]
dt1[, `Transit Code` := NULL]

# Remove commas from numeric fields stored as character
dt1[, `Actual Weight in Tons`      := as.numeric(gsub(",", "", `Actual Weight in Tons`))]
dt1[, `Freight Revenue ($)`        := as.numeric(gsub(",", "", `Freight Revenue ($)`))]
dt1[, `Estimated Short Line Miles` := as.numeric(gsub(",", "", `Estimated Short Line Miles`))]
dt1[, `Expanded Carloads`          := as.numeric(gsub(",", "", `Expanded Carloads`))]
dt1[, `Expanded Tons`              := as.numeric(gsub(",", "", `Expanded Tons`))]
dt1[, `Expanded Freight Revenue`   := as.numeric(gsub(",", "", `Expanded Freight Revenue`))]


# =============================================================================
# 6. ENCODE CATEGORICAL VARIABLES AS FACTORS
# =============================================================================

dt1[, `STB Car Type`                    := as.factor(`STB Car Type`)]
dt1[, `STCC2`                           := as.factor(`STCC2`)]
dt1[, `Origin Freight Rate Territory`   := as.factor(`Origin Freight Rate Territory`)]
dt1[, `TerminationFreightRateTerritory` := as.factor(`TerminationFreightRateTerritory`)]
dt1[, `Car Ownership Code`              := as.factor(`Car Ownership Code`)]
dt1[, `Waybill Month`                   := as.factor(`Waybill Month`)]


# =============================================================================
# 7. ADJUST FREIGHT REVENUE FOR INFLATION
# Annual PPI averages for railroad freight (Base: Dec 1996 = 100)
# =============================================================================

ppi_lookup <- data.table(
  `Data Year` = 2005:2024,
  PPI = c(123.53, 135.90, 140.92, 157.28, 148.48,
          156.23, 169.80, 177.46, 183.09, 186.51,
          179.48, 175.46, 181.85, 192.08, 197.85,
          197.88, 207.50, 226.88, 231.65, 236.24)
)

# 2024 is the base year (most recent complete year in dataset)
ppi_2024 <- 236.24

dt1 <- merge(dt1, ppi_lookup, by = "Data Year", all.x = TRUE)
dt1[, `Freight Revenue Real` := `Freight Revenue ($)` * (ppi_2024 / PPI)]
dt1[, PPI := NULL]


# =============================================================================
# 8. ADDITIONAL DATA CLEANING
# =============================================================================

# Drop rows with zero or negative values in key fields
dt1 <- dt1[`Estimated Short Line Miles` > 0]
dt1 <- dt1[`Freight Revenue ($)` > 0]
dt1 <- dt1[`Actual Weight in Tons` > 0]

# Derived feature: weight per carload (reduces multicollinearity)
dt1[, weight_per_car := `Actual Weight in Tons` / `Number of Carloads`]


# =============================================================================
# 9. MODEL 1: BASELINE LINEAR REGRESSION (full dataset sample)
# =============================================================================

set.seed(42)
dt_sample <- dt1[sample(.N, 200000)]

model1 <- lm(`Freight Revenue Real` ~
  `Number of Carloads` +
  `Actual Weight in Tons` +
  `Estimated Short Line Miles` +
  `Number of Interchanges` +
  `Car Ownership Code` +
  `STB Car Type` +
  `STCC2` +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Waybill Month` +
  `Type of Move (inferred)`,
  data = dt_sample)

summary(model1)
# R^2 = 0.74, but residual standard error = ~$48,930 (very large)
# Cone-shaped residuals indicate heteroscedasticity -> try log transform

# Backward stepwise selection by AIC
step(model1)
# Result: Waybill Month is dropped; all other predictors retained

# Check multicollinearity
library(car)
vif(model1)
# STB Car Type and STCC2 have high raw GVIF, but GVIF^(1/(2*Df)) is acceptable


# =============================================================================
# 10. MODEL 2: LOG-LINEAR REGRESSION (log outcome, partial log predictors)
# =============================================================================

set.seed(42)
dt_sample <- dt1[sample(.N, 200000)]

model2 <- lm(log(`Freight Revenue Real`) ~
  log(`Number of Carloads`) +
  log(`Estimated Short Line Miles` + 1) +
  `Number of Interchanges` +
  `weight_per_car` +
  `Car Ownership Code` +
  `STB Car Type` +
  `STCC2` +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Type of Move (inferred)`,
  data = dt_sample)

summary(model2)
# R^2 = 0.8976, residual SE = 0.4656 (log scale ~+/-59% in dollar terms)


# =============================================================================
# 11. MODEL 3: LOG-LINEAR WITH log(weight_per_car)
# =============================================================================

set.seed(42)
dt_sample <- dt1[sample(.N, 200000)]

model3 <- lm(log(`Freight Revenue Real`) ~
  log(`Number of Carloads`) +
  log(`Estimated Short Line Miles`) +
  `Number of Interchanges` +
  log(weight_per_car) +
  `Car Ownership Code` +
  `STB Car Type` +
  `STCC2` +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Type of Move (inferred)`,
  data = dt_sample)

summary(model3)
# R^2 = 0.9036, residual SE = 0.4443 (log scale ~+/-55%)

# Visualize residuals
plot(model3)

# Test for heteroscedasticity (note: very sensitive at n=200,000)
ncvTest(model3)

# Example prediction with smearing correction for log-normal bias
new_shipment <- data.frame(
  `Number of Carloads`                = 1,
  `Estimated Short Line Miles`        = 500,
  `Number of Interchanges`            = 1,
  weight_per_car                      = 80,
  `Car Ownership Code`                = factor("P", levels = levels(dt_sample$`Car Ownership Code`)),
  `STB Car Type`                      = factor("51", levels = levels(dt_sample$`STB Car Type`)),
  `STCC2`                             = factor("28", levels = levels(dt_sample$STCC2)),
  `All Rail/Intermodal Code`          = factor("All Rail", levels = levels(dt_sample$`All Rail/Intermodal Code`)),
  `Hazardous/Bulk Material in Boxcar` = factor("H", levels = levels(dt_sample$`Hazardous/Bulk Material in Boxcar`)),
  `Origin Freight Rate Territory`     = factor("4", levels = levels(dt_sample$`Origin Freight Rate Territory`)),
  `TerminationFreightRateTerritory`   = factor("2", levels = levels(dt_sample$`TerminationFreightRateTerritory`)),
  `Type of Move (inferred)`           = 0,
  check.names = FALSE
)

pred <- predict(model3, newdata = new_shipment, interval = "prediction", level = 0.95)
# Apply smearing correction: multiply by exp(0.5 * sigma^2)
exp(pred) * exp(0.5 * 0.4443^2)
# fit ~$6,876, 95% PI: [$2,878, $16,429]

# Investigate outliers
outliers <- which(abs(rstandard(model3)) > 3)
length(outliers)
dt_sample[outliers, .(`Freight Revenue Real`, `Number of Carloads`, `Estimated Short Line Miles`, STCC2)]


# =============================================================================
# 12. EXPLORATORY ANALYSIS
# =============================================================================

# Median real freight revenue by year
yearly_median <- dt1[, .(median_rev = median(`Freight Revenue Real`, na.rm = TRUE)), by = `Data Year`][order(`Data Year`)]
print(yearly_median)
# Notable: large jump in 2021 due to (1) COVID supply chain surge and
# (2) STB expanded waybill sampling effective Jan 1, 2021

plot(yearly_median$`Data Year`, yearly_median$median_rev,
  type = "b",
  main = "Median Real Freight Revenue by Year",
  xlab = "Year",
  ylab = "Median Revenue (2024 $)")

# Entries per year
year_counts <- table(dt1$`Data Year`)
barplot(year_counts,
  main = "Number of Entries per Year",
  xlab = "Year",
  ylab = "Number of Entries",
  las = 2)

# Distribution of carload counts (capped at 50)
carload_counts_capped <- table(dt1[`Number of Carloads` <= 50]$`Number of Carloads`)
barplot(carload_counts_capped,
  main = "Number of Entries by Carload Count (up to 50)",
  xlab = "Number of Carloads",
  ylab = "Number of Entries",
  las = 2)

# Multi-car shipments by year
dt1[, .(
  multi_car_count = sum(`Number of Carloads` > 1),
  total = .N,
  percent_multicar = round(100 * sum(`Number of Carloads` > 1) / .N, 2)
), by = `Data Year`][order(`Data Year`)]


# =============================================================================
# 13. POST-2020 MODELS
# Data collection methodology changed Jan 1, 2021 (STB rule expansion)
# Revenue levels also shifted due to COVID supply chain disruption
# =============================================================================

dt_2021 <- dt1[`Data Year` >= 2021]
nrow(dt_2021)  # ~3.66 million rows

set.seed(42)
all_idx   <- sample(nrow(dt_2021))
train_idx <- all_idx[1:100000]
test_idx  <- all_idx[100001:200000]

dt_sample1 <- dt_2021[train_idx]
dt_test    <- dt_2021[test_idx]

# Model 11: linear (post-2021 data)
model11 <- lm(`Freight Revenue Real` ~
  `Number of Carloads` +
  weight_per_car +
  `Estimated Short Line Miles` +
  `Number of Interchanges` +
  `Car Ownership Code` +
  `STB Car Type` +
  `STCC2` +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Waybill Month` +
  `Type of Move (inferred)`,
  data = dt_sample1)

summary(model11)
# R^2 = 0.748, residual SE = ~$31,070

step(model11)
# Waybill Month dropped again

# Model 13: log-linear (post-2021 data)
model13 <- lm(log(`Freight Revenue Real`) ~
  log(`Number of Carloads`) +
  log(`Estimated Short Line Miles`) +
  `Number of Interchanges` +
  log(weight_per_car) +
  `Car Ownership Code` +
  `STB Car Type` +
  `STCC2` +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Type of Move (inferred)`,
  data = dt_sample1)

summary(model13)
# R^2 = 0.8683, residual SE = 0.4119 (~+/-51%)

plot(model13)


# =============================================================================
# 14. MODEL 15: SHORT-HAUL DUMMY + INTERACTION (no quadratic terms)
# First-order model introducing short_haul as a factor. Used as a baseline
# to test whether adding quadratic terms (model16) significantly improves fit.
# =============================================================================

# Create short-haul dummy (< 500 miles)
dt_sample1[, short_haul := factor(as.integer(`Estimated Short Line Miles` < 500))]
dt_test[,    short_haul := factor(as.integer(`Estimated Short Line Miles` < 500))]

model15 <- lm(log(`Freight Revenue Real`) ~
  log(`Estimated Short Line Miles`) +
  short_haul +
  log(`Number of Carloads`) +
  log(weight_per_car) +
  `Number of Interchanges` +
  `Car Ownership Code` +
  `STB Car Type` +
  STCC2 +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Type of Move (inferred)` +
  short_haul:`Number of Interchanges`,
  data = dt_sample1)

summary(model15)


# =============================================================================
# 15. MODEL 16: QUADRATIC TERMS + SHORT-HAUL INTERACTION
# Short-haul rail pricing is structurally different from long-haul:
# terminal handling costs are fixed, making per-mile rates much higher
# =============================================================================

# Create short-haul dummy (threshold chosen based on outlier analysis)
dt_sample1[, short_haul := factor(as.integer(`Estimated Short Line Miles` < 500))]
dt_test[,    short_haul := factor(as.integer(`Estimated Short Line Miles` < 500))]

model16 <- lm(log(`Freight Revenue Real`) ~
  log(`Estimated Short Line Miles`) +
  short_haul +
  log(`Number of Carloads`) +
  log(weight_per_car) +
  `Number of Interchanges` +
  `Car Ownership Code` +
  `STB Car Type` +
  STCC2 +
  `All Rail/Intermodal Code` +
  `Hazardous/Bulk Material in Boxcar` +
  `Origin Freight Rate Territory` +
  `TerminationFreightRateTerritory` +
  `Type of Move (inferred)` +
  I(log(`Estimated Short Line Miles`)^2) +
  I(log(`Number of Carloads`)^2) +
  I(log(weight_per_car)^2) +
  short_haul:`Number of Interchanges`,
  data = dt_sample1)

summary(model16)
# R^2 = 0.8725, residual SE = 0.4054

# Outlier count
outliers16 <- which(abs(rstandard(model16)) > 3)
length(outliers16)  # 1,162

# ANOVA: test whether quadratic terms in model16 significantly improve on model15
anova(model15, model16)
# p-value is very small -> retain model16 (quadratic terms are justified)

# =============================================================================
# 15. EVALUATE MODEL 16 ON HELD-OUT TEST SET
# =============================================================================

# Align factor levels
dt_test[, STCC2 := factor(STCC2, levels = levels(dt_sample1$STCC2))]

pred_test <- predict(model16, newdata = dt_test)
actual    <- log(dt_test$`Freight Revenue Real`)

rmse   <- sqrt(mean((pred_test - actual)^2, na.rm = TRUE))
ss_res <- sum((actual - pred_test)^2, na.rm = TRUE)
ss_tot <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
r2_test <- 1 - ss_res / ss_tot

cat("Test RMSE:", rmse, "\n")       # 0.404
cat("Test R²:",  r2_test, "\n")     # 0.875
cat("Train R²:", summary(model16)$r.squared, "\n")  # 0.876

# Dollar-level accuracy (with smearing correction)
pred_dollars   <- exp(pred_test) * exp(0.5 * 0.4054^2)
actual_dollars <- dt_test$`Freight Revenue Real`

within_15 <- mean(abs(pred_dollars - actual_dollars) / actual_dollars <= 0.15, na.rm = TRUE) * 100
cat("Predictions within 15% of actual:", round(within_15, 1), "%\n")
# Result: 32.9% — reflects fundamental noise from masked contract rates
