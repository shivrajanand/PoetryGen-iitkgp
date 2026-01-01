import torch
import transformers
import argparse
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import utils
from datetime import datetime
import os
import json
import bitsandbytes as bnb
from functools import partial


# To add a new dataset, you need to add it's tags in this dict
# specify any arguments to load_dataaset in the special `load_kwargs` key
dataset_lang_tags_map = {
    "anushtup": {"English": "English", "Sanskrit": "Sanskrit"},
    "rahular/itihasa": {"English": "en", "Sanskrit": "sn"},
    "mitrasamgraha": {"English": "english", "Sanskrit": "sanskrit_Deva"},
    "mitrasamgraha_metered": {"English": "english", "Sanskrit": "sanskrit_Deva"},
    "OOD_test_df": {"English": "engtrans_input"},
}


all_models = [
    "ai4bharat/IndicBART",
    "facebook/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-1.3B",
    "ai4bharat/indictrans2-en-indic-dist-200M",
    "CohereForAI/aya-23-8B",
    "sanganaka/nllb-200-distilled-1.3B-FT_mitrasamgraha"
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, help="Input dataset, in hf datasets format"
    )
    parser.add_argument(
        "--model_hf",
        type=str,
        default="ai4bharat/IndicBART",
        help="hf transformers model name",
        choices=all_models,
    )
    parser.add_argument("-B", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--full_ft", default=False, action='store_true', help="Whether to use full fine tuning (false=lora by default)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora_rslora",
        action="store_true",
        default=False,
        help="Use rank-stabilized LoRA?",
    )
    parser.add_argument(
        "--lora_dora", action="store_true", default=False, help="Use DoRA?"
    )
    parser.add_argument(
        "--predict_model",
        type=str,
        help="If given model dir, will only perform prediction and not training",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        help="If given model ckpt dir, will continue training from there",
    )
    parser.add_argument("--special_tokens", action='store_true', default=False, help="Use special tokens in training?")

    args = parser.parse_args()
    valid_datasets = "\n".join(dataset_lang_tags_map.keys())
    assert (
        args.dataset in dataset_lang_tags_map
    ), f"Dataset {args.dataset} does not have language tags, add them in `dataset_lang_tags_map`.\nCurrently supported datasets are {valid_datasets}"

    if args.predict_model is not None:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.predict_model)
        expdir = os.path.dirname(args.predict_model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.predict_model)
        tokenizer.tgt_lang = 'san_Deva'
        with open(f'{expdir}/config.json') as f:
            trained_config = json.load(f)
        args.model_hf = trained_config['_name_or_path']
        print(f"Evaluating pretrained {args.model_hf} located at {expdir}")
    else:
        if args.model_hf == "ai4bharat/IndicBART":
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_hf)
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_hf)
        elif args.model_hf in ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "sanganaka/nllb-200-distilled-1.3B-FT_mitrasamgraha"]:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_hf, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
            tokenizer_name = args.model_hf if args.model_hf != "sanganaka/nllb-200-distilled-1.3B-FT_mitrasamgraha" else "facebook/nllb-200-distilled-1.3B"
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
            tokenizer.tgt_lang = 'san_Deva'
            if args.special_tokens:
                tokenizer.add_special_tokens({
                    'additional_special_tokens': tokenizer.additional_special_tokens + ['<anushtup>']
                })
        # elif args.model_hf == 'saucam/Rudra-7b':
        #     model = transformers.AutoModelForCausalLM.from_pretrained(args.model_hf)
        elif args.model_hf == "ai4bharat/indictrans2-en-indic-dist-200M":
            from IndicTransTokenizer import IndicProcessor

            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                args.model_hf, trust_remote_code=True
            )
            ip = IndicProcessor(inference=False)
            utils.ip = ip
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model_hf, trust_remote_code=True
            )
        elif args.model_hf == "CohereForAI/aya-23-8B":
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model_hf,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_hf)

        if not args.full_ft:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                use_rslora=args.lora_rslora,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_dora=args.lora_dora,
                target_modules=[
                    "self_attn.q_proj",
                    "self_attn.v_proj",
                    "encoder_attn.q_proj",
                    "encoder_attn.v_proj",
                ],
            )
            model = get_peft_model(model, peft_config)

            print("Model stats:")
            model.print_trainable_parameters()
        expdir = f"trained_models/{args.model_hf.replace('/', '.')}/{datetime.now().strftime('%d-%mT%H.%M')}"
        os.makedirs(expdir)
    print("Model:", expdir)

    # pad each batch with a custom collate_fn

    # Data loading
    utils.tokenizer = tokenizer
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    # if args.hf_hub_data is None:
    #     dataset = load_dataset(
    #         "csv", data_files=args.train_data, header=0, split="train"
    #     )
    #     dataset = dataset.train_test_split(test_size=0.1, shuffle=False)
    #     lang_names = {"English": "English", "Sanskrit": "Sanskrit"}
    # else:
    lang_names = dataset_lang_tags_map[args.dataset]
    dataset = load_dataset(args.dataset, **lang_names.get('load_kwargs', {}))
    utils.lang_names = lang_names
    dataset = utils.filter_dataset(dataset, args.dataset)
    preprocess = partial(utils.preprocess, hf_name=args.model_hf, data_name=args.dataset, special_tokens=args.special_tokens)

    dataset_tkn = dataset.map(preprocess, batched=True, num_proc=8)

    train_dataset = dataset_tkn["train"]
    test_dataset = dataset_tkn["test"]

    # Taken from https://huggingface.co/docs/transformers/tasks/translation

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=expdir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4 * args.batch_size,
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        report_to="none",
        generation_max_length=256,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics,
    )

    if args.predict_model is None:
        print("Start of train".center(80, "="))
        trainer.train(args.resume_ckpt)

        os.makedirs(expdir + "/final")
        trainer.save_model(expdir + "/final")

    # import ipdb; ipdb.set_trace()

    # trainer.evaluate()
    preds, labels, metrics = trainer.predict(
        test_dataset, metric_key_prefix="test", max_length=256
    )
    print("Final metrics:", metrics)
    dataset_safe_name = args.dataset.replace("/", "-")
    with open(f"{expdir}/{dataset_safe_name}_final_results.json", 'w') as f:
        json.dump(metrics, f)

    if utils.ip is None:
        decoded_preds = utils.postprocess(preds)
    else:
        decoded_preds = utils.postprocess_indictrans(preds)

    predictions_data = utils.get_predictions_data(test_dataset, decoded_preds)

    predictions_data.to_csv(f"{expdir}/{dataset_safe_name}_generated_poetry.csv")