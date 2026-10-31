# Fig. 3 representative-seed selection after T=1000-aligned recalculation

Seed 0 is reserved for Fig. 2 and excluded from the Fig. 3 candidate pool.
The original core-three rules are reapplied without using WGAN-GP:

1. Diffusion-dominant: smallest Diffusion-only mean absolute error across five metrics.
2. NMM-favorable: largest mean advantage `Diffusion-only error - NMM-only error` at ACF lags 0.5 s and 1.0 s, excluding an already selected seed.
3. Median-like: closest overall mean absolute error to the candidate-pool median, excluding the model-specific cases.
4. Exact numerical ties are resolved by the lower seed number.

Candidate-pool median overall mean absolute error: `0.0659505868823863`.

- Fig. 3(a), median-like: seed `3`
- Fig. 3(b), diffusion-dominant: seed `2`
- Fig. 3(c), NMM-favorable at longer lags: seed `16`

The seed identifiers are reproducibility metadata; the manuscript may retain criterion-based panel descriptions.
