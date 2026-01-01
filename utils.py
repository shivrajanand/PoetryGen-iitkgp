import numpy as np
import evaluate
import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


tokenizer = None
ip = None
lang_names = None
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")


def preprocess(data, hf_name:str, data_name:str, special_tokens=False):
    """Preprocess function for huggingface dataset (to be mapped)
    
    Args:
        data: The input data loaded by datasets.load_dataset
        hf_name (str): Model name
    
    Returns:
        Preprocessed inputs
    """
    # <2{lang_id}> is from https://huggingface.co/ai4bharat/IndicBART (and in general for MBart)
    assert tokenizer is not None
    assert lang_names is not None

    model_inputs = None
    labels = None

    # now using pre-transliterated data
    # if data_name == "mitrasamgraha":
    #     for i,sample in enumerate(data[lang_names["Sanskrit"]]):
    #         data[lang_names["Sanskrit"]][i] = transliterate(sample, sanscript.IAST, sanscript.DEVANAGARI)

    if hf_name == "ai4bharat/indictrans2-en-indic-dist-200M":
        assert ip is not None
        inputs = ip.preprocess_batch(
            data[lang_names["English"]],
            src_lang="eng_Latn",
            tgt_lang="san_Deva",
        )
        targets = data[lang_names["Sanskrit"]]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, truncation=True)
    elif hf_name == "CohereForAI/aya-23-8B":
        messages = get_aya23_message_format(data[lang_names["English"]])
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            return_tensors="pt",
        )
        targets = data[lang_names["Sanskrit"]]
    elif hf_name.startswith("facebook/nllb-200-distilled-"):
        # inputs = [f"eng_Latn {sample} </s>" for sample in data[lang_names["English"]]]
        model_inputs = tokenizer(data[lang_names["English"]])
        # targets = [f"san_Deva {sample} </s>" for sample in data[lang_names["Sanskrit"]]]
        targets = data[lang_names["Sanskrit"]]
        if special_tokens:
            targets = [
                '<anushtup>' + x if c.startswith('anuṣṭubh') and 'asamīcīna' not in c 
                else x
                for c,x in zip(data['chanda'], data[lang_names["Sanskrit"]])
            ]
        labels = tokenizer(text_target=targets)
    else:
        inputs = [f"{sample} </s> <2en>" for sample in data[lang_names["English"]]]
        targets = [f"<2sa> {sample} </s>" for sample in data[lang_names["Sanskrit"]]]

    if not model_inputs:
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)

    # with tokenizer.as_target_tokenizer():
    if not labels:
        labels = tokenizer(targets, max_length=256, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Taken from https://huggingface.co/CohereForAI/aya-23-8B/blob/main/Aya_23_notebook.ipynb
def get_aya23_message_format(prompts):
    return [{"role": "user", "content": p} for p in prompts]


# Taken from https://huggingface.co/docs/transformers/tasks/translation
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    assert tokenizer is not None
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # with tokenizer.as_target_tokenizer():
    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result_bleu = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result_chrf = chrf.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result_bleu["score"], "chrf": result_chrf["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def postprocess(preds):
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds


def postprocess_indictrans(preds):
    print("USING INDICTRANS POSTPROC")
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    with tokenizer.as_target_tokenizer():
        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    decoded_preds = [pred.strip() for pred in decoded_preds]
    # decoded_preds = ip.postprocess_batch(decoded_preds, lang='san_Deva')

    return decoded_preds


def get_predictions_data(test_dataset, decoded_preds):
    assert lang_names is not None
    return pd.DataFrame(
        {
            "english_INPUT": test_dataset[lang_names["English"]],
            "sanskrit_PRED": decoded_preds,
            "sanskrit_GT": test_dataset[lang_names["Sanskrit"]],
        }
    )


def filter_dataset(dataset, dataset_name):
    if dataset_name == "rahular/itihasa":

        def itihasa_mapper(batch):
            english, sanskrit = [], []
            for sample in batch["translation"]:
                english.append(sample[lang_names["English"]])
                sanskrit.append(sample[lang_names["Sanskrit"]])
            return {lang_names["English"]: english, lang_names["Sanskrit"]: sanskrit}

        dataset = dataset.map(itihasa_mapper, batched=True, num_proc=8).select_columns(
            list(lang_names.values())
        )
    return dataset


def count_tokens(outputs):
    """Counts the number of tokens in each output sample, excluding padding.

    Args:
        outputs: A tensor of shape (batch_size, sequence_length) containing padded sequences.

    Returns:
        A list of integers, representing number of tokens in the corresponding output sample.
    """

    token_counts = []
    for output in outputs:
        padding_index = (output == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]

        # If padding is found, count tokens up to the padding index, otherwise use the full length.
        if padding_index.nelement() > 0:
            token_count = padding_index[0].item()
        else:
            token_count = output.shape[0]

        token_counts.append(token_count)
    return token_counts