import torch
from dataclasses import dataclass
from skrutable.meter_identification import MeterIdentifier, VerseTester
from skrutable.scansion import Scanner, Verse
from skrutable.meter_patterns import anuzwuB_pAda
from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
from datetime import datetime
from datasets import Dataset
import pandas as pd
from ft_sanskrit import dataset_lang_tags_map

# ! Matches all strings with anuṣṭubh in the pure/vipula forms, not asamīcīna, not as sub-part of another (upajati)
valid_label_regex = re.compile(r'^anuṣṭubh \(\d,\d\: (?!asamīcīna).+?, \d,\d\: (?!asamīcīna).+?\)$')


@dataclass
class MetricsOutput:
    name: str
    meter_verses: List[Verse]

    histogram_lengths: Dict[int, int]
    histogram_labels: Dict[str, int]

    semantic_similarities: torch.Tensor


def evaluate_generated(inputs, poetry_outputs: List[str], dataset_name: str = ''):
    """
    :param inputs: hindi meaning (ground truth)
    :param poetry_outputs: Description (model output)
    :type poetry_outputs: List[str]
    :param dataset_name: Description
    :type dataset_name: str chandomitra: default
    """
    histogram_labels, histogram_lengths, meter_verses = make_anushtup_histograms(poetry_outputs)

    semantic_model = SentenceTransformer('sanganaka/bge-m3-sanskritFT')
    in_embs = semantic_model.encode(inputs, convert_to_tensor=True)
    out_embs = semantic_model.encode(poetry_outputs, convert_to_tensor=True)

    sims = semantic_model.similarity_pairwise(in_embs, out_embs)
    # print(sims)
    # import ipdb; ipdb.set_trace()
    return MetricsOutput(
        name=f"{dataset_name}-{datetime.now()}",
        meter_verses=meter_verses,
        histogram_lengths=histogram_lengths,
        histogram_labels=histogram_labels,
        semantic_similarities=sims,
    )


def make_anushtup_histograms(poetry_outputs: List[str]):
    """
    Docstring for make_anushtup_histograms
    
    :param poetry_outputs: 
    :type poetry_outputs: List[str]
    """
    mi = MeterIdentifier()
    meter_verses = []
    for x in poetry_outputs:
        i = 0
        while i < len(x):
            try:
                # first x character is a matra/other rarer edge cases
                meter = mi.identify_meter(x[i:], from_scheme='DEV', resplit_option='resplit_max')
            except:
                i += 1
                continue
            meter_verses.append(meter)
            break
        if i == len(x):
            # nothing found in while loop, happens for 1 sample in the test set (total 1421).
            meter = mi.identify_meter('')
            meter_verses.append(meter)

    histogram_lengths = {}
    histogram_labels = {}
    for x in meter_verses:
        sw = x.syllable_weights.replace('\n','')
        histogram_lengths[len(sw)] = histogram_lengths.get(len(sw), 0) + 1
        histogram_labels[x.meter_label] = histogram_labels.get(x.meter_label, 0) + 1
    # print(histogram_lengths)
    # print(histogram_labels)
    return histogram_labels, histogram_lengths, meter_verses


def calculate_anushtup_percentages(
        histogram_labels: Dict[str, int], 
        histogram_lengths: Dict[int, int]
    ) -> Tuple[float, float] :
    total_count = 0
    anushtup_count = 0


    for label, count in histogram_labels.items():
        total_count += count
        if valid_label_regex.match(label) is not None:
            anushtup_count += count

    assert total_count != 0, "histogram_labels has a total count of 0."
    assert total_count == sum(histogram_lengths.values()), "total counts of the two histograms are different"

    full_anushtup_percent = anushtup_count * 100 / total_count

    full_length_count = histogram_lengths.get(32, 0)
    partial_anushtup_percent = (full_length_count - anushtup_count) * 100 / total_count

    return full_anushtup_percent, partial_anushtup_percent


def cosine_sim_to_percentage(cos_sim: List[float], reduce=True) -> float:
    # Converts the sinusoidal distances b/w multiple pairs to a linear percentage
    dists = ((torch.pi - cos_sim.arccos()) * 100 / torch.pi)
    if reduce:
        dists = dists.mean()
    return dists


def save_outputs(dataset: Dataset, dataset_name: str, outputs: List[str], metrics: MetricsOutput) -> pd.DataFrame:
    
    names = dataset_lang_tags_map[dataset_name]
    
    anushtup_type = []
    for x in metrics.meter_verses:
        a_t = "None"
        if valid_label_regex.match(x.meter_label) is not None:
            a_t = "Full"
        elif len(x.syllable_weights.replace('\n','')) == 32:
            a_t = "Partial"
        anushtup_type.append(a_t)

    df = pd.DataFrame()
    df['Inputs'] = dataset[names['English']]
    df['GT'] = dataset[names['Sanskrit']]
    df['Outputs'] = outputs
    df['anushtup_type'] = anushtup_type
    df['semantic_sim %'] = cosine_sim_to_percentage(metrics.semantic_similarities, reduce=False).cpu()
    
    return df