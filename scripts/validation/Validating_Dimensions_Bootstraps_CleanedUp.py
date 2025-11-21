# -*- coding: utf-8 -*-
"""
Cross-validate the four stigma dimensions on bootstrapped models and report
cosine similarities across dimensions.

This version removes hardcoded Windows paths. Run it as a module from the
repository root, for example:

    python -m scripts.validation.Validating_Dimensions_Bootstraps_CleanedUp \
        --modeling-dir /path/to/BootstrappedModels \
        --lexicon-path /path/to/Stigma_WordLists.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from ..lexicon import build_lexicon_stigma, dimension_stigma
from ..common.path_config import add_path_arguments, build_path_config


YEARS = [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
DEFAULT_MODEL_PREFIX = "CBOW_300d__win10_min50_iter3"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate dimension stability on bootstrapped models.")
    add_path_arguments(parser, require_raw_data_root=False)
    parser.add_argument(
        "--model-prefix",
        default=DEFAULT_MODEL_PREFIX,
        help="Prefix used when loading bootstrapped models (before year and boot suffix).",
    )
    parser.add_argument(
        "--boot-number",
        type=int,
        default=0,
        help="Which bootstrapped model number to evaluate for each window.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=YEARS,
        help="Three-year window start years to process.",
    )
    return parser.parse_args()


def load_dimension_terms(paths):
    lexicon = pd.read_csv(paths.lexicon_path)
    lexicon = lexicon[lexicon["Removed"] != "remove"]

    dangerouswords = lexicon.loc[(lexicon["WhichPole"] == "dangerous")]["Term"].str.lower().tolist()
    safewords = lexicon.loc[(lexicon["WhichPole"] == "safe")]["Term"].str.lower().tolist()

    disgustingwords = lexicon.loc[(lexicon["WhichPole"] == "disgusting")]["Term"].str.lower().tolist()
    enticingwords = lexicon.loc[(lexicon["WhichPole"] == "enticing")]["Term"].str.lower().tolist()

    purewords = lexicon.loc[(lexicon["WhichPole"] == "pure")]["Term"].str.lower().tolist()
    impurewords = lexicon.loc[(lexicon["WhichPole"] == "impure")]["Term"].str.lower().tolist()

    traits = pd.read_csv(paths.personality_traits_path)
    traits["Adjective"] = traits["Adjective"].str.lower().str.strip()
    traits = traits.drop_duplicates(subset="Adjective")
    negwords = traits[traits["Sentiment"] == "neg"]["Adjective"].tolist()
    poswords = traits[traits["Sentiment"] == "pos"]["Adjective"].tolist()

    return {
        "dangerous": (dangerouswords, safewords),
        "disgust": (disgustingwords, enticingwords),
        "purity": (impurewords, purewords),
        "negpos": (negwords, poswords),
    }


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    term_pairs = load_dimension_terms(paths)

    trackers = {
        "train_accuracy_N_danger": [],
        "train_accuracy_percent_danger": [],
        "holdout_accuracy_N_danger": [],
        "holdout_accuracy_percent_danger": [],
        "train_accuracy_N_disgust": [],
        "train_accuracy_percent_disgust": [],
        "holdout_accuracy_N_disgust": [],
        "holdout_accuracy_percent_disgust": [],
        "train_accuracy_N_purity": [],
        "train_accuracy_percent_purity": [],
        "holdout_accuracy_N_purity": [],
        "holdout_accuracy_percent_purity": [],
        "train_accuracy_N_negpos": [],
        "train_accuracy_percent_negpos": [],
        "holdout_accuracy_N_negpos": [],
        "holdout_accuracy_percent_negpos": [],
        "cossim_pure_danger": [],
        "cossim_pure_disgust": [],
        "cossim_pure_negpos": [],
        "cossim_danger_disgust": [],
        "cossim_negpos_disgust": [],
        "cossim_negpos_danger": [],
        "mostsim_danger": [],
        "leastsim_danger": [],
        "mostsim_purity": [],
        "leastsim_purity": [],
        "mostsim_negpos": [],
        "leastsim_negpos": [],
        "mostsim_disgust": [],
        "leastsim_disgust": [],
    }

    for yr1 in args.years:
        yr3 = yr1 + 2
        print(f"PROCESSING MODEL FOR YEAR: {yr1}")
        model_path = paths.bootstrap_model_path(yr1, args.boot_number, args.model_prefix)
        currentmodel = Word2Vec.load(str(model_path))

        dangerwords = build_lexicon_stigma.dimension_lexicon(currentmodel, *term_pairs["dangerous"])
        disgustwords = build_lexicon_stigma.dimension_lexicon(currentmodel, *term_pairs["disgust"])
        puritywords = build_lexicon_stigma.dimension_lexicon(currentmodel, *term_pairs["purity"])
        negposwords = build_lexicon_stigma.dimension_lexicon(currentmodel, *term_pairs["negpos"])

        danger = dimension_stigma.dimension(dangerwords, "larsen")
        disgust = dimension_stigma.dimension(disgustwords, "larsen")
        purity = dimension_stigma.dimension(puritywords, "larsen")
        negpos = dimension_stigma.dimension(negposwords, "larsen")

        dimtemp = dimension_stigma.kfold_dim(disgustwords)
        trackers["train_accuracy_N_disgust"].append(dimtemp[0])
        trackers["train_accuracy_percent_disgust"].append(dimtemp[1])
        trackers["holdout_accuracy_N_disgust"].append(dimtemp[2])
        trackers["holdout_accuracy_percent_disgust"].append(dimtemp[3])

        dimtemp = dimension_stigma.kfold_dim(puritywords)
        trackers["train_accuracy_N_purity"].append(dimtemp[0])
        trackers["train_accuracy_percent_purity"].append(dimtemp[1])
        trackers["holdout_accuracy_N_purity"].append(dimtemp[2])
        trackers["holdout_accuracy_percent_purity"].append(dimtemp[3])

        dimtemp = dimension_stigma.kfold_dim(dangerwords)
        trackers["train_accuracy_N_danger"].append(dimtemp[0])
        trackers["train_accuracy_percent_danger"].append(dimtemp[1])
        trackers["holdout_accuracy_N_danger"].append(dimtemp[2])
        trackers["holdout_accuracy_percent_danger"].append(dimtemp[3])

        dimtemp = dimension_stigma.kfold_dim(negposwords)
        trackers["train_accuracy_N_negpos"].append(dimtemp[0])
        trackers["train_accuracy_percent_negpos"].append(dimtemp[1])
        trackers["holdout_accuracy_N_negpos"].append(dimtemp[2])
        trackers["holdout_accuracy_percent_negpos"].append(dimtemp[3])

        trackers["cossim_pure_danger"].append(cosine_similarity(purity.dimensionvec().reshape(1, -1), danger.dimensionvec().reshape(1, -1)))
        trackers["cossim_pure_disgust"].append(cosine_similarity(purity.dimensionvec().reshape(1, -1), disgust.dimensionvec().reshape(1, -1)))
        trackers["cossim_pure_negpos"].append(cosine_similarity(purity.dimensionvec().reshape(1, -1), negpos.dimensionvec().reshape(1, -1)))
        trackers["cossim_danger_disgust"].append(cosine_similarity(danger.dimensionvec().reshape(1, -1), disgust.dimensionvec().reshape(1, -1)))
        trackers["cossim_negpos_disgust"].append(cosine_similarity(negpos.dimensionvec().reshape(1, -1), disgust.dimensionvec().reshape(1, -1)))
        trackers["cossim_negpos_danger"].append(cosine_similarity(negpos.dimensionvec().reshape(1, -1), danger.dimensionvec().reshape(1, -1)))

        trackers["mostsim_danger"].append(currentmodel.wv.similar_by_vector(danger.dimensionvec(), topn=10))
        trackers["leastsim_danger"].append(currentmodel.wv.similar_by_vector(-danger.dimensionvec(), topn=10))
        trackers["mostsim_purity"].append(currentmodel.wv.similar_by_vector(purity.dimensionvec(), topn=10))
        trackers["leastsim_purity"].append(currentmodel.wv.similar_by_vector(-purity.dimensionvec(), topn=10))
        trackers["mostsim_disgust"].append(currentmodel.wv.similar_by_vector(disgust.dimensionvec(), topn=10))
        trackers["leastsim_disgust"].append(currentmodel.wv.similar_by_vector(-disgust.dimensionvec(), topn=10))
        trackers["mostsim_negpos"].append(currentmodel.wv.similar_by_vector(negpos.dimensionvec(), topn=10))
        trackers["leastsim_negpos"].append(currentmodel.wv.similar_by_vector(-negpos.dimensionvec(), topn=10))

    for key in [
        "cossim_pure_danger",
        "cossim_pure_disgust",
        "cossim_pure_negpos",
        "cossim_danger_disgust",
        "cossim_negpos_disgust",
        "cossim_negpos_danger",
    ]:
        values = trackers[key]
        print(round(np.mean(values), 2))
        print(round(np.std(values), 2))
        print("next")


if __name__ == "__main__":
    main()
