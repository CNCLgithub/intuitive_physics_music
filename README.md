# Model and analysis code for the paper for "An Intuitive Physics Approach to Modeling Melodic Expectation"

Melodic continuation is modeled as a physical system in which a note is pulled by three "musical forces" adapted from Larson's theory of melodic forces: **gravity**, **magnetism**, and **inertia**. This is implemented as a generative probabilistic program (the **PMFM**, Probabilistic Melodic Forces Model) in [Gen.jl](https://www.gen.dev/), with melodic completions inferred via particle filtering and compared against human melodic-expectation judgments and competing computational models (IDyOM, Temperley).

## Directory structure

```
Code/
├── PMFM.ipynb                  # Main model: generative definition, particle-filter inference,
│                                #   model-vs-human correlations, per-melody visualizations
├── Data_Processing.ipynb       # Builds data/ CSVs from raw MIDI stems + human response data (OSF)
├── analyses.Rmd                # Main-text + supplemental statistical analyses (model comparisons,
│                                #   bootstrapped R², relative-importance regression, contour/cadence effects)
├── Revision_Analyses.ipynb     # Context-sensitivity analysis: PMFM vs. IDyOM on repeated melodic
│                                #   fragments with differing prior context
├── Supplemental_Analyses.ipynb # Parameter grid search, single-force ablations (no gravity / no
│                                #   magnetism / no inertia), log-likelihood analyses, per-melody plots
├── pmfm_fragment_ppd.jl        # Standalone script computing PMFM's posterior predictive distribution
│                                #   for the fragment queries used in Revision_Analyses.ipynb
├── utils/                      # Julia modules shared by the notebooks/scripts
│   ├── scale.jl                #   Note / MusicalScale types, scale construction
│   ├── force.jl                #   calculate_gravity / calculate_magnetism / calculate_inertia
│   ├── data.jl                 #   Loading melodies/rhythms/keys, building observation DataFrames
│   ├── draw.jl                 #   Custom Gen.jl `draw` distribution (categorical draw from a list)
│   └── truncatednorm.jl        #   Custom Gen.jl truncated-normal distribution
├── MusicalExpectation/         # Julia project environment (Project.toml / Manifest.toml) activated
│                                #   by the notebooks/scripts via `Pkg.activate("MusicalExpectation")`
├── data/                       # Input data: melody stems (MIDI + CSV), human response data, and
│                                #   precomputed predictions from comparison models (IDyOM, Temperley HP)
├── exports/                    # CSV/PNG outputs written by the notebooks (correlations, ablations,
│                                #   grid-search results, fragment PPD results, per-model plots)
└── plots/                      # Additional figures (particle histograms, bootstrap R² plots, etc.)
```

## Pipeline

1. **`Data_Processing.ipynb`** (Python) parses the raw MIDI melody stems in `data/midi/` and the human
   completion-judgment data (`data/exp2_data.csv`, from the Morgan et al. (2019) dataset on
   [OSF](https://osf.io/wgz9t/)) into the tidy CSVs in `data/` and `exports/` used downstream
   (`all_melodies_final_means.csv`, `aggregate_human_responses.csv`, `individual_responses.csv`, etc.).
2. **`PMFM.ipynb`** (Julia) defines the generative model (`melody` / `melody_kernel`), runs particle-filter
   inference over each melody stem, and correlates model predictions against human mean completions,
   producing `exports/correlation_full.csv` and the `exports/pf_exports/` per-melody trace plots.
3. **`analyses.Rmd`** (R) consumes the exported model and human data (plus precomputed baselines from
   IDyOM and Temperley Hansen-Pearce in `data/`) to run the main-text and supplemental statistics:
   correlation/R² comparisons, bootstrapped confidence intervals, relative-importance regression, and
   melodic contour/cadence effects.
4. **`Revision_Analyses.ipynb`** (Python) builds paired fragment-context queries from the melody stems,
   writes them to `data/fragment_context_queries.json`, and — after running
   `julia --project=MusicalExpectation pmfm_fragment_ppd.jl` to produce PMFM's posterior predictive
   distributions — compares PMFM's and IDyOM's context-sensitivity against ground truth.
5. **`Supplemental_Analyses.ipynb`** (Julia) runs the parameter grid search, single-force ablations, and
   log-likelihood analyses reported in the supplement, plus per-melody plots in `exports/melody_plots/`.

## Requirements

- **Julia** (notebooks/scripts using `Pkg.activate("MusicalExpectation")`): [Gen.jl](https://www.gen.dev/),
  Plots, CSV, DataFrames, Distributions, HypothesisTests, GLM, ProgressBars, JSON — see
  `MusicalExpectation/Project.toml` for the pinned environment.
- **Python** (`Data_Processing.ipynb`, `Revision_Analyses.ipynb`): pandas, numpy, scipy, music21,
  matplotlib.
- **R** (`analyses.Rmd`): dplyr, tidyr, tidyverse, stringr, ggplot2, boot, ggprism, ggpubr, relaimpo,
  gridExtra, effsize.

## Data sources

- Melody stems and human completion judgments: Morgan, E., et al. (2019), [osf.io/wgz9t](https://osf.io/wgz9t/).
- Authentic/Non-authentic cadence stimulus preprocessing replicated from [osf.io/47hdb](https://osf.io/47hdb).
- Comparison model predictions (IDyOM, Temperley Hansen-Pearce) precomputed and included under `data/`.
