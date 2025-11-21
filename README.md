# disease_stigma

**This code repository accompanies the paper "[The Stigma of Diseases: Unequal Burden, Uneven Decline](https://osf.io/preprints/socarxiv/7nm9x/)" by Rachel Kahn Best and Alina Arseniev-Koehler, forthcoming in _American Sociological Review_.** 

Preprint of paper is available here: https://osf.io/preprints/socarxiv/7nm9x/. Details and context for this code are described in the paper and appendices. 

**Paper abstract:** Why are some diseases more stigmatized than others? And, has disease stigma declined over time? Answers to these questions have been hampered by a lack of comparable data. Using word embedding methods, we analyze 4.7 million news articles to create new measures of stigma for 106 health conditions from 1980-2018. Using mixed effects regressions, we find that behavioral health conditions and preventable diseases attract the strongest connotations of immorality and negative personality traits. Meanwhile, infectious diseases are marked by disgust. These results lend new empirical support to theories that norm enforcement and contagion avoidance drive disease stigma. Challenging existing theories, we find no evidence for a link between medicalization and stigma, and inconclusive evidence on the relationship between advocacy and stigma. Finally, we find that stigma has declined dramatically over time, but only for chronic physical illnesses. In the past four decades, stigma has transformed from a sea of negative connotations surrounding most diseases to two primary conduits of meaning: infectious diseases spark disgust, and behavioral health conditions cue negative stereotypes. These results show that cultural meanings are especially durable when they are anchored by interests, and that cultural changes intertwine in ways that only become visible through large-scale research.



## Repository layout

The repository is organized by function under the `scripts/` directory; shared inputs live in `data/`.

* `scripts/corpus/prepare_corpus_from_csv.py`: Convert CSV corpora into yearly pickled article lists.
* `scripts/training/TrainingPhraser_CleanedUp.py`: Train phrase models on 3-year windows.
* `scripts/training/TrainingW2V_Booted_CleanedUp.py`: Train bootstrapped Word2Vec models for each window.
* `scripts/scoring/WriteStigmaScores_CleanedUp.py`: Compute dimension scores for each disease across bootstraps.
* `scripts/scoring/AggregatingBootstraps_CleanedUp.py`: Aggregate dimension scores across bootstraps.
* `scripts/scoring/AggregatingStigmaScores_StigmaIndex_CleanedUp.py`: Aggregate stigma index scores and confidence intervals.
* `scripts/scoring/WordCounts_CleanedUp.py`: Count disease mentions across bootstrapped models.
* `scripts/analysis/PlottingBootstrapped_CleanedUp.py`: Plot aggregated results.
* `scripts/validation/Validating_OverallW2VModels_CleanedUp.py`: Evaluate trained Word2Vec models on WordSim and analogy tests.
* `scripts/validation/Validating_Dimensions_Bootstraps_CleanedUp.py`: Validate stability of the four stigma dimensions across bootstraps.
* `scripts/lexicon/build_lexicon_stigma.py` and `scripts/lexicon/dimension_stigma.py`: Helpers for building lexicons and semantic dimensions.

Key data files (all under `data/`):

* `Final_Search_SymptomsDiseasesList.txt`: Search terms used to collect LexisNexis articles (raw data not redistributed).
* `Disease_list_5.12.20_uncorrupted.csv`: Disease names and plotting groups.
* `Stigma_WordLists.csv`: Stigma lexicon with dimension poles.
* `updated_personality_trait_list.csv`: Personality trait list for the neg/pos dimension.
* `questions_words_pasted.txt`: Analogy benchmark questions for Word2Vec validation.
* `stigmaindex_aggregated_temp_92CI.csv`: Example aggregated scores for plotting.

## Running with configurable paths

All scripts now accept shared path arguments so you can point to your own corpus, models, and results directories without editing code. Key options:

* `--raw-data-root`: Base directory containing the `NData_<year>` folders with raw article pickles. Optional when a script does not read raw data.
* `--contemp-data-root`: Optional base for `ContempData_<year>` folders; defaults to `--raw-data-root` when omitted.
* `--modeling-dir`: Location to read or write modeling artifacts (bigrams, bootstraps, embeddings). Optional for scripts that only read aggregated outputs.
* `--results-dir`: Destination for result CSVs and plots; defaults to the current working directory.
* `--analyses-dir`: Base directory for imports and ancillary files; defaults to the current working directory.
* `--lexicon-path`, `--disease-list-path`, `--personality-traits-path`: Paths to the supporting CSV inputs, defaulting to the copies in this repository.

Each script requests only the paths it needs. For example, `scripts/scoring/AggregatingStigmaScores_StigmaIndex_CleanedUp.py` requires `--modeling-dir` and `--results-dir`, while `scripts/analysis/PlottingBootstrapped_CleanedUp.py` only needs `--results-dir` unless you override input filenames. Run the scripts as modules from the repository root (e.g., `python -m scripts.scoring.WriteStigmaScores_CleanedUp --help`).

**scripts/validation/Validating_OverallW2VModels_CleanedUp.py**
* Validate an overall word2vec model on the WordSim-353 test and Google analogy test.
* Requires: `data/questions_words_pasted.txt` (override with `--analogy-file`).

**scripts/validation/Validating_Dimensions_Bootstraps_CleanedUp.py**
* Cross-validation for each of 4 dimensions and cosine similarities between dimensions.
* Requires: `scripts/lexicon/build_lexicon_stigma.py`, `scripts/lexicon/dimension_stigma.py`, and the lexicon/personality trait CSVs.
* Note: We do not include code or data for comparing our dimensions to human-rated data collected by Pachankis et. al; we cannot distribute data from Pachankis et al.

**scripts/scoring/WriteStigmaScores_CleanedUp.py**
* Compute each of 4 stigma scores and medicalization score for each disease, in each model, in each time period. Write results to CSVs (one CSV per dimension, per time window).
* Requires: `scripts/lexicon/build_lexicon_stigma.py`, `scripts/lexicon/dimension_stigma.py`, `data/updated_personality_trait_list.csv`, `data/Stigma_WordLists.csv`, and `data/Disease_list_5.12.20_uncorrupted.csv`.

**scripts/scoring/AggregatingStigmaScores_StigmaIndex_CleanedUp.py**
* Aggregate bootstrapped scores for the time windows and 4 dimensions to get a mean and 92% confidence interval for each disease's mean loading across the 4 dimensions (i.e., stigma score) in each time window. Write results to a single CSV (this CSV is also included in `data/stigmaindex_aggregated_temp_92CI.csv`).

**scripts/scoring/AggregatingBootstraps_CleanedUp.py**
* Aggregate bootstrapped scores for time windows to get a mean and 92% confidence interval for each disease's loading on a dimension in a given time window. Write results to a CSV (one CSV per dimension).

**scripts/scoring/WordCounts_CleanedUp.py**
* Compute the number of mentions for each disease, in each model, in each time period. Get a mean and 92% confidence interval for each disease's number of mentions in a given time window. Write per-year CSVs plus an aggregated CSV.

**scripts/analysis/PlottingBootstrapped_CleanedUp.py**
* Visualize stigma scores of diseases, by disease group, across time. (Requires `stigmaindex_aggregated_temp_92CI.csv` unless you override `--input-file`).

## Using your own corpus

The training scripts expect pickled article lists organized as `NData_<year>/all<year>bodytexts_regexeddisamb_listofarticles`
under a `--raw-data-root` directory. If your data are in a CSV (for example, with
`title`, `Text`, and `Date` columns), you can generate the expected pickles with
`scripts/corpus/prepare_corpus_from_csv.py`:

```bash
python -m scripts.corpus.prepare_corpus_from_csv \
  --csv-path /path/to/your/news.csv \
  --text-column Text \
  --title-column title \
  --date-column Date \
  --default-year 2010 \  # optional fallback if dates are missing
  --id-column GOID \     # optional: drop duplicates with this column
  --min-body-chars 50 \  # optional: skip very short rows
  --encoding utf-8 \     # optional: override CSV encoding
  --output-basename articles_{year}.pkl \  # optional: shorter filenames
  --write-manifest \       # optional: write manifest.json describing processing steps
  --output-root /path/to/raw_data_root
```

The script will create one pickle per year in `output-root`, matching the
layout expected by `scripts/training/TrainingPhraser_CleanedUp.py` and
`scripts/training/TrainingW2V_Booted_CleanedUp.py`. If you override `--output-basename`, keep
`{year}` in the template so each folder still contains a distinct file per year;
`--write-manifest` adds a `manifest.json` per year to document the cleaning and
splitting steps without encoding that history in the filename.
