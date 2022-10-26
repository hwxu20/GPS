#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reproduce the main evaluation in `Multitask Prompted Training Enables Zero-Shot Task Generalization` using PyTorch.

This script is heavily adapted from https://github.com/huggingface/transformers/blob/7533d30acd975027e83a548e4c38e06fa335291b/examples/pytorch/multiple-choice/run_swag_no_trainer.py
"""

import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import json
import codecs

import datasets
import torch
from datasets import load_dataset, load_metric, concatenate_datasets, Value
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import Counter

import transformers
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    default_data_collator,
)
from transformers.file_utils import PaddingStrategy
# from promptsource.templates import DatasetTemplates
from templates import DatasetTemplates

logger = logging.getLogger(__name__)


def read_split_list(file_name):
    test_task_list = []
    with codecs.open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            task_tuple = line.split('/')
            if len(task_tuple) == 2:
                test_task_list.append(task_tuple)
            else:
                test_task_list.append((task_tuple[0], None))

    return test_task_list


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    # parser.add_argument("--dataset_name", type=str, default=None,
    #                     help="The name of the dataset to use (via the datasets library).", required=True, )
    # parser.add_argument(
    #     "--dataset_config_name",
    #     type=str,
    #     default=None,
    #     help="The configuration name of the dataset to use (via the datasets library).",
    # )
    # parser.add_argument("--template_name", type=str, default=None,
    #                     help="The template/prompt name", required=True, )
    parser.add_argument("--max_length", type=int, default=1024,
                        help=(
                            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."),
                        )
    parser.add_argument("--target_max_length", type=int, default=256,
                        help="Target max length. Sequences longer than this will be truncated." )
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
                        )
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
                        required=True, )
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument("--template_dir", type=str, default='/localdata/codebook/t-zero/templates_test',
                        help="模版文件的位置", )
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).", )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate debug mode and run training only with a subset of data.", )
    parser.add_argument("--ga_dev_distribution", type=str, choices=['uniform', 'ratio'], default='uniform',
                        help="ga_dev的分布", )
    parser.add_argument("--parallelize", action="store_true",
                        help=(
                            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
                            "Note that this feature is still experimental in HF Transformers."),
                        )
    parser.add_argument("--test_split", type=str, help='测试任务名单')
    parser.add_argument("--dataset_type", type=str, choices=['ga', 'all'])
    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


def build_ga_dataset(dataset_name, dataset_config_name, raw_datasets, distribution='uniform'):
    task_name = f'{dataset_name}/{dataset_config_name}' if dataset_config_name else f'{dataset_name}'
    dataset_distribution = {'anli/r1': {0: 334, 2: 333, 1: 333},
                            'anli/r2': {0: 334, 1: 333, 2: 333},
                            'anli/r3': {0: 402, 1: 402, 2: 396},
                            'super_glue/cb': {1: 28, 0: 23, 2: 5},
                            'super_glue/rte': {0: 146, 1: 131},
                            'super_glue/wsc.fixed': {0: 66, 1: 38},
                            'winogrande/winogrande_xl': {'2': 639, '1': 628},
                            'super_glue/copa': {0: 55, 1: 45},
                            'hellaswag': {'2': 2584, '0': 2515, '1': 2485, '3': 2458},
                            'super_glue/wic': {0: 319, 1: 319},
                            'story_cloze/2016': {1: 962, 2: 909}
                            }

    label_key = 'label'
    if dataset_name in ['winogrande']:
        label_key = 'answer'
    if dataset_name in ['story_cloze']:
        label_key = 'answer_right_ending'
    filtered_dataset = raw_datasets

    # 对anli数据集和winogrande的特别处理
    if dataset_name == 'anli':
        # print(f'len of raw_dataset: {filtered_dataset}')
        filtered_dataset = filtered_dataset.filter(lambda x: len(x['reason']) > 0, load_from_cache_file=False)
        # print(f'len of filtered_dataset: {filtered_dataset}')
        if dataset_config_name == 'r1':
            filtered_dataset = filtered_dataset.select(range(900))
        elif dataset_config_name == 'r2':
            filtered_dataset = filtered_dataset.select(range(1500))
        elif dataset_config_name == 'r3':
            index = list(range(len(filtered_dataset)))
            index = index[8000:]
            filtered_dataset = filtered_dataset.select(index)
    if dataset_name == 'winogrande':
        # filtered_dataset = load_dataset('winogrande', 'winogrande_debiased', split='train')
        train_file_path = '/localdata/codebook/data/T0_dataset'
        train_file_path = os.path.join(train_file_path, 'winogrande_winogrande_debiased')
        data_files = {
            'train': os.path.join(train_file_path, 'train.json'),
        }
        filtered_dataset = load_dataset('json', data_files=data_files)['train']

    label_list = filtered_dataset[label_key]
    label_type_set = set(label_list)
    print(f'label_type_set: {label_type_set}')
    ga_dataset_list = []
    # 把每个类别的样本分类存储
    for label_type in label_type_set:
        single_label_dataset = filtered_dataset.filter(lambda x: x[label_key] == label_type, load_from_cache_file=False)
        single_label_dataset = single_label_dataset.shuffle(seed=42)

        if distribution == 'ratio':
            example_num_per_label = math.ceil(
                dataset_distribution[task_name][label_type] / sum(dataset_distribution[task_name].values()) * 32)
        else:
            example_num_per_label = math.ceil(32 / len(label_type_set))
        ga_dataset_list.append(single_label_dataset.select(
            range(min(example_num_per_label, len(single_label_dataset)))))

    filtered_dataset = concatenate_datasets(ga_dataset_list)

    return filtered_dataset


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, )
    logger.info(accelerator.state)

    # Metrics
    metric = load_metric("accuracy")

    test_task_list = read_split_list(args.test_split)
    logger.info(f'测试任务: {test_task_list}')

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        # T0 model
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    if args.parallelize:
        assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
        model.parallelize()
    else:
        model.to(device)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    for dataset_name, dataset_config_name in test_task_list:
        train_file_path = '/localdata/codebook/data/T0_dataset'
        if dataset_config_name:
            train_file_path = os.path.join(train_file_path, f'{dataset_name}_{dataset_config_name}')
        else:
            train_file_path = os.path.join(train_file_path, dataset_name)
        data_files = {
            'train': os.path.join(train_file_path, 'train.json'),
            'validation': os.path.join(train_file_path, 'validation.json')
        }
        if args.dataset_type == 'ga' or args.dataset_type == 'pt':
            if dataset_name == 'story_cloze':
                raw_datasets = load_dataset('json', data_files=data_files)['validation']
            else:
                raw_datasets = load_dataset('json', data_files=data_files)['train']
        else:
            raw_datasets = load_dataset('json', data_files=data_files)['validation']

        if dataset_name == 'story_cloze':
            new_feature = raw_datasets.features.copy()
            new_feature['answer_right_ending'] = Value('int64', id=None)
            raw_datasets = raw_datasets.cast(new_feature, load_from_cache_file=False)

        logger.info(f'raw dataset for {dataset_name}_{dataset_config_name}: {raw_datasets}')

        # Trim a number of evaluation examples
        if args.debug:
            raw_datasets = raw_datasets.select(range(min(len(raw_datasets), 100)))
        elif args.dataset_type == 'ga':
            pass  # 不截断
        else:
            # raw_datasets = raw_datasets.select(range(1000))
            if dataset_name != 'hellaswag':
                raw_datasets = raw_datasets.select(range(min(len(raw_datasets), 1000)))

        column_names = raw_datasets.column_names
        logger.info(f'column name: {column_names}')

        def preprocess_function(examples):
            bs = len(examples[column_names[0]])

            input_texts = []
            target_texts = []
            answer_choices_texts = []
            for i in range(bs):
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                # print(f'debug: ex: {ex}')
                input, target = template.apply(ex)
                # print(f'debug: input: {input}')
                # print(f'debug: target: {target}')

                ex_answer_choices = template.get_answer_choices_list(ex)
                assert target in ex_answer_choices
                input_texts.append(input)
                target_texts.append(target)
                answer_choices_texts.append(ex_answer_choices)

            tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            tokenized_targets = [
                tokenizer(
                    ans_choi,
                    padding=True,
                    max_length=args.target_max_length,
                    truncation=True,
                )
                for ans_choi in answer_choices_texts
            ]

            features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

            features["labels"] = [
                tokenized_targets[idx]["input_ids"]
                for idx in range(bs)
            ]
            features["labels_attention_mask"] = [
                tokenized_targets[idx]["attention_mask"]
                for idx in range(bs)
            ]
            features["targets"] = [
                answer_choices_texts[idx].index(t)
                for idx, t in enumerate(target_texts)
            ]

            return features

        # Get the prompt to apply and the possible targets.
        logger.info(f'use template_dir: {args.template_dir}')
        # if dataset_name == 'anli' and args.dataset_type != 'ga':
        #     # ga时r1 r2 r3视作不同的数据集
        #     prompts = DatasetTemplates(dataset_name, template_dir=args.template_dir)
        # else:
        #     prompts = DatasetTemplates(
        #         f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
        #         template_dir=args.template_dir)
        prompts = DatasetTemplates(
            f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
            template_dir=args.template_dir)

        # key是uuid
        # template_list = prompts.name_to_id_mapping.keys()
        template_list = prompts.templates.keys()
        logger.info(f'{dataset_name}的模板列表：{template_list}')

        result_summary = []
        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name

            logger.info(f'{template.metadata.original_task}, type: {type(template.metadata.original_task)}')
            if template.metadata.original_task is not True:
                logger.info(f'跳过{template_name}, 因为不是原始任务形式')
                continue

            prediction_list = []

            # 过滤copa样本, 一些prompt只适用于部分样本
            filtered_dataset = None
            if dataset_config_name == 'copa':
                if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'effect', load_from_cache_file=False)
                if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
                    filtered_dataset = raw_datasets.filter(lambda example: example['question'] == 'cause', load_from_cache_file=False)

            # ga的32 dev在这里过滤，不然filter完可能少于32
            if args.dataset_type == 'ga':
                if not filtered_dataset:
                    filtered_dataset = raw_datasets
                filtered_dataset = build_ga_dataset(dataset_name, dataset_config_name, filtered_dataset,
                                                    args.ga_dev_distribution)
                label_key = 'label'
                if dataset_name in ['winogrande']:
                    label_key = 'answer'
                if dataset_name in ['story_cloze']:
                    label_key = 'answer_right_ending'

                print(f'filtered_dataset: {filtered_dataset}')
                print(f'label distribution: {Counter(filtered_dataset[label_key])}')

            print(f'evaluating {dataset_name}_{dataset_config_name}_{template_name}')

            with accelerator.main_process_first():
                if filtered_dataset:
                    eval_dataset = filtered_dataset.map(
                        preprocess_function, batched=True, remove_columns=column_names,
                        load_from_cache_file=False)
                else:
                    eval_dataset = raw_datasets.map(
                        preprocess_function, batched=True, remove_columns=column_names,
                        load_from_cache_file=False)

            # Log a few random samples from the eval set:
            # 随机展示几个样本
            if args.debug:
                for index in random.sample(range(len(eval_dataset)), 3):
                    logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

            # DataLoaders creation:
            if args.pad_to_max_length:
                # If padding was already done ot max length, we use the default data collator that will just convert everything
                # to tensors.
                data_collator = default_data_collator
            else:
                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                data_collator = DataCollatorForMultipleChoice(
                    tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                )

            eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
                                         batch_size=args.per_device_eval_batch_size)

            # Prepare everything with our `accelerator`.
            eval_dataloader = accelerator.prepare(eval_dataloader)

            # Eval!
            total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes

            logger.info("***** Running evaluation *****")
            logger.info(f"  Num examples = {len(eval_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
            logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

            model.eval()
            for batch in eval_dataloader:
                # batch: attention_mask, input_ids, labels(答案的token ids), labels_attention_mask
                model_inputs = {
                    k: batch[k]
                    for k in ["input_ids", "attention_mask", "labels"]
                }
                with torch.no_grad():
                    # [batch_size, seq_len, vocab]
                    logits = model(**model_inputs).logits

                # [batch_size, seq_len, vocab]
                masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
                # [batch_size, seq_len]
                seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
                # [batch_size, ]
                seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
                seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
                # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
                # [batch_size, choice_num]
                predictions = seq_log_prob.argmax(dim=-1)

                exam_num = seq_log_prob.shape[0]
                choice_num = seq_log_prob.shape[1]
                # 记录预测结果
                for exam_id, batch_idx in enumerate(range(0, exam_num*choice_num, choice_num)):
                    prediction_result = {
                        'input_text:': tokenizer.decode(batch['input_ids'][batch_idx], skip_special_tokens=True),
                        'target_text': tokenizer.decode(batch['labels'][batch_idx], skip_special_tokens=True),
                        'label:': batch['targets'][exam_id].item(),
                        'prediction:': predictions[exam_id].item(),
                        'logits': seq_log_prob[exam_id].cpu().numpy().tolist()
                    }
                    prediction_list.append(prediction_result)

                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["targets"]),
                )

                progress_bar.update(1)

            eval_metric = metric.compute()
            accelerator.print(f"Result: {eval_metric}")

            results = {
                "dataset_name": dataset_name,
                "dataset_config_name": dataset_config_name,
                "template_id": template.get_id(),
                "template_name": template_name,
                "evaluation": eval_metric
            }
            result_summary.append(results)
            # 记录每条预测结果
            prediction_dir = os.path.join(args.output_dir, 'predictions')
            os.makedirs(prediction_dir, exist_ok=True)
            if accelerator.is_main_process:
                output_name = '_'.join(dataset_name.split(' '))
                if dataset_config_name is not None:
                    output_name = output_name + '_' + '_'.join(dataset_config_name.split(' '))
                if template_name.find('/') != -1:
                    output_name = output_name + '_' + '_'.join(template_name.split('/'))
                else:
                    output_name = output_name + '_' + '_'.join(template_name.split(' '))
                with open(os.path.join(prediction_dir, f"{output_name}.json"), "w") as f:
                    json.dump(prediction_list, f, indent=4)

        # 写入当前数据集的acc
        if accelerator.is_main_process:
            output_name = '_'.join(dataset_name.split(' '))
            if dataset_config_name is not None:
                output_name = output_name + '_' + '_'.join(dataset_config_name.split(' '))
            with open(os.path.join(args.output_dir, f"{output_name}.json"), "w") as f:
                json.dump(result_summary, f, indent=4)


if __name__ == "__main__":
    main()
