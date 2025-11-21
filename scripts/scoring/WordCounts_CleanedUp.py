# -*- coding: utf-8 -*-
"""
Compute bootstrapped word counts for each disease and aggregate across years.
"""

import argparse
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize

from ..common.path_config import add_path_arguments, build_path_config


YEARS = [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
DEFAULT_BOOT_RANGE = range(25)
DEFAULT_MODEL_PREFIX = "CBOW_300d__win10_min50_iter3"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Count disease mentions across bootstrapped models.")
    add_path_arguments(parser, require_raw_data_root=False)
    parser.add_argument(
        "--model-prefix",
        default=DEFAULT_MODEL_PREFIX,
        help="Prefix used when loading bootstrapped Word2Vec models.",
    )
    parser.add_argument(
        "--boot-range",
        type=int,
        nargs="+",
        default=list(DEFAULT_BOOT_RANGE),
        help="Bootstrapped model numbers to process (default: 0-24).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store per-year and aggregated CSVs (defaults to --results-dir).",
    )
    return parser.parse_args()


def fold_word(target, second, wvmodel):
    weight_target = wvmodel.wv.vocab[target].count / (wvmodel.wv.vocab[target].count + wvmodel.wv.vocab[second].count)
    weight_second = wvmodel.wv.vocab[second].count / (wvmodel.wv.vocab[target].count + wvmodel.wv.vocab[second].count)
    weighted_wv = (weight_target * normalize(wvmodel.wv[target].reshape(1, -1))) + (weight_second * normalize(wvmodel.wv[second].reshape(1, -1)))
    return normalize(weighted_wv)


def add_folded_terms(model):
    epilepsy_folded = fold_word("epilepsy", "epileptic", model)
    drug_addiction_folded = fold_word("drug_addiction", "drug_addict", model)
    obesity_folded = fold_word("obesity", "obese", model)

    model.wv.add("epilepsy_folded", epilepsy_folded)
    model.wv["drug_addiction_folded"] = drug_addiction_folded
    model.wv["obesity_folded"] = obesity_folded

    model.wv.vocab["epilepsy_folded"].count = model.wv.vocab["epileptic"].count + model.wv.vocab["epilepsy"].count
    model.wv.vocab["drug_addiction_folded"].count = model.wv.vocab["drug_addict"].count + model.wv.vocab["drug_addiction"].count
    model.wv.vocab["obesity_folded"].count = model.wv.vocab["obese"].count + model.wv.vocab["obesity"].count

    return model


def load_diseases(paths):
    diseases = pd.read_csv(paths.disease_list_path)
    diseases = diseases[diseases["Plot"] == "Yes"]
    diseases = diseases.drop_duplicates(subset=["Reconciled_Name"])
    return diseases[["PlottingGroup", "Reconciled_Name"]].copy()


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    boot_range = list(args.boot_range)
    output_dir = args.output_dir or paths.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_year_files = []
    base_diseases = load_diseases(paths)

    for yr1 in YEARS:
        diseases = base_diseases.copy()
        yr3 = yr1 + 2
        for bootnum in boot_range:
            model_path = paths.bootstrap_model_path(yr1, bootnum, args.model_prefix)
            currentmodel = Word2Vec.load(str(model_path))
            currentmodel = add_folded_terms(currentmodel)

            diseases[f"wordcount_{bootnum}"] = diseases["Reconciled_Name"].apply(
                lambda x: currentmodel.wv.vocab[str(x).lower()].count if str(x).lower() in currentmodel.wv.vocab else "NA"
            )

        diseases["Year"] = [str(yr1)] * len(diseases)
        wordcount_cols = [f"wordcount_{boot}" for boot in boot_range]
        melted = pd.melt(
            diseases[["Reconciled_Name", "PlottingGroup", "Year", *wordcount_cols]],
            id_vars=["Reconciled_Name", "PlottingGroup", "Year"],
            var_name="BootNumber",
            value_name="wordcount",
        )

        output_file = output_dir / f"wordcount_{yr1}.csv"
        melted.to_csv(output_file, index=False)
        per_year_files.append(output_file)

    combined = pd.concat((pd.read_csv(file) for file in per_year_files))
    combined.to_csv(output_dir / "wordcount_aggregated.csv", index=False)


if __name__ == "__main__":
    main()
