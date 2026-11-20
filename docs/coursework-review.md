# Coursework review

**Module:** Fundamentals of Data Analytics, M.Sc. Data Analytics, BSBI, Term 1 (submitted July 2024)
**Brief:** three tasks. Retail sales with linear algebra and regression, production quality with ANOVA or t-tests, and card fraud with vectors, matrices and anomaly detection.

## Why I generated the data this time

| Task | Data in the original | Problem | This version |
|---|---|---|---|
| 1. Sales | An e-commerce order file from India (128,975 orders, 91 days) with a marketing spend column added | One region only, three months only, and marketing spend was not part of the source data. Seasonality and regional patterns could not be studied, and the marketing effect could not be checked. | Four regions, 36 months, planted trend, season and marketing effect |
| 2. Quality | 300 production records | No way to know which effects were real | 600 batches with planted effects, including one factor (line) with no effect |
| 3. Fraud | 1,003 transactions with card numbers and no fraud label | A classifier (logistic regression) was described, but without labels it cannot be trained or scored. Card numbers were in the file. | Unsupervised detectors, labels kept in a separate file for scoring only, no card numbers |

## What changed in the analysis

**Task 1.** The original regression reported R² = 0.61 and read the marketing coefficient as cause and effect ("2 more units of marketing give 2.5 units of sales"). The rebuild solves the same kind of regression three ways to show the linear algebra (normal equations, QR, SVD), checks the condition number, adds an SVD of the region x month matrix, and tests the forecast on held-out months. It also shows why the marketing coefficient should not be read as causal: in the generated data it comes out at 1.8 against a planted 2.4, because marketing spend moves with region size.

**Task 2.** Testing factors one at a time is what the brief suggested, and the rebuild shows where that goes wrong: one false alarm at 5% (the line) and one missed effect (the shift). Effect size, a Bonferroni correction and one regression with all factors fix both.

**Task 3.** The brief asked for anomaly detection, which needs no labels. The rebuild compares three detectors (robust z-score, Mahalanobis distance, Isolation Forest) and measures them against planted fraud. It sets the review queue size first, because that is what a fraud team works with.
