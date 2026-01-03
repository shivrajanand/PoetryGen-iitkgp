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
import sys
import time
from datetime import timedelta


# ! Matches all strings with anuṣṭubh in the pure/vipula forms, not asamīcīna, not as sub-part of another (upajati)
valid_label_regex = re.compile(r'^anuṣṭubh \(\d,\d\: (?!asamīcīna).+?, \d,\d\: (?!asamīcīna).+?\)$')


@dataclass
class MetricsOutput:
    name: str
    meter_verses: List[Verse]

    histogram_lengths: Dict[int, int]
    histogram_labels: Dict[str, int]

    semantic_similarities: torch.Tensor


def evaluate_generated(inputs, 
                       poetry_outputs: List[str], 
                       dataset_name: str = '', 
                       semantic_model: SentenceTransformer | None = None,):

    histogram_labels, histogram_lengths, meter_verses = make_anushtup_histograms(poetry_outputs)

    # semantic_model = SentenceTransformer('sanganaka/bge-m3-sanskritFT')
    # in_embs = semantic_model.encode(inputs, convert_to_tensor=True)
    # out_embs = semantic_model.encode(poetry_outputs, convert_to_tensor=True)
    
    # Above three lines are redesigned to use gpu and become faster
    if semantic_model is None:
        semantic_model = SentenceTransformer(
            'sanganaka/bge-m3-sanskritFT',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    with torch.inference_mode():
        in_embs = semantic_model.encode(
            inputs,
            convert_to_tensor=True,
            device=semantic_model.device
        )
        out_embs = semantic_model.encode(
            poetry_outputs,
            convert_to_tensor=True,
            device=semantic_model.device
        )


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


############## CUSTOM FUNCTION ADDED TO ACCEPT CSV #############################

incsv = sys.argv[1]
df = pd.read_csv(incsv)
print(f"Loaded {len(df)} samples")

# Check if required columns exist
assert 'meaning' in df.columns, "dataframe must have 'meaning' column which are hindi meanings of sanskrit verses"
assert 'model-out' in df.columns, "dataframe must have 'model-out' column which is the sanskrit verses generate by the model"

# Extract Sanskrit outputs
meaning = df['meaning'].tolist()
model_out = df['model-out'].tolist()

# Evaluate the Sanskrit outputs
metrics = evaluate_generated(meaning, model_out, dataset_name='translations_output_imp')

# Calculate anushtubh percentages
full_anushtup_percent, partial_anushtup_percent = calculate_anushtup_percentages(
    metrics.histogram_labels, 
    metrics.histogram_lengths
)

# Calculate none percentage
none_percent = 100 - full_anushtup_percent - partial_anushtup_percent

# Calculate semantic similarity
semantic_sim = cosine_sim_to_percentage(metrics.semantic_similarities)

# Print results
print(f"\n{'='*60}")
print(f"EVALUATION RESULTS")
print(f"{'='*60}")
print(f"Full Anushtubh:    {full_anushtup_percent:.2f}%")
print(f"Partial Anushtubh: {partial_anushtup_percent:.2f}%")
print(f"None:              {none_percent:.2f}%")
print(f"\nSemantic Similarity: {semantic_sim:.2f}%")
print(f"{'='*60}")

# Print detailed histograms
print(f"\nHistogram of Syllable Lengths:")
for length in sorted(metrics.histogram_lengths.keys()):
    count = metrics.histogram_lengths[length]
    percentage = count * 100 / len(model_out)
    print(f"  Length {length:2d}: {count:4d} samples ({percentage:5.2f}%)")

print(f"\nHistogram of Meter Labels (top 10):")
sorted_labels = sorted(metrics.histogram_labels.items(), key=lambda x: x[1], reverse=True)
for i, (label, count) in enumerate(sorted_labels[:10]):
    percentage = count * 100 / len(model_out)
    print(f"  {i+1:2d}. {label[:50]:50s}: {count:4d} ({percentage:5.2f}%)")

# Save detailed results
anushtup_type = []
for x in metrics.meter_verses:
    a_t = "None"
    if valid_label_regex.match(x.meter_label) is not None:
        a_t = "Full"
    elif len(x.syllable_weights.replace('\n','')) == 32:
        a_t = "Partial"
    anushtup_type.append(a_t)

results_df = df.copy()
results_df['anushtup_type'] = anushtup_type
results_df['semantic_sim_%'] = cosine_sim_to_percentage(metrics.semantic_similarities, reduce=False).cpu()
results_df['meter_label'] = [x.meter_label for x in metrics.meter_verses]
results_df['syllable_count'] = [len(x.syllable_weights.replace('\n','')) for x in metrics.meter_verses]

output_filename = incsv.replace("Generate-Poetry/", "/Generate-Poetry/Metrics/")
results_df.to_csv(output_filename, index=False)
print(f"\nDetailed results saved to: {output_filename}")