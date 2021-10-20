# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" k-adapter for OpenEntity"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaTokenizerFast

from pytorch_transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaTokenizer,
    WarmupLinearSchedule,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)
from pytorch_transformers.modeling_roberta import gelu
from pytorch_transformers.my_modeling_roberta import RobertaForEntityTyping
from examples.utils_glue import (
    convert_examples_to_features_ner,
    output_modes,
    processors,
)

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForEntityTyping, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    pretrained_model = model[0]
    ner_model = model[1]

    # if args.local_rank in [-1, 0]:
    # tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    if args.freeze_bert:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in ner_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in ner_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in ner_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in ner_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        if args.freeze_bert:
            ner_model, optimizer = amp.initialize(ner_model, optimizer, opt_level=args.fp16_opt_level)
        else:
            ner_model, optimizer = amp.initialize(ner_model, optimizer, opt_level=args.fp16_opt_level)
            pretrained_model, optimizer = amp.initialize(pretrained_model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.freeze_bert:
            ner_model = torch.nn.DataParallel(ner_model)
        else:
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            ner_model = torch.nn.DataParallel(ner_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.freeze_bert:
            ner_model = torch.nn.parallel.DistributedDataParallel(
                ner_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )
        else:
            ner_model = torch.nn.parallel.DistributedDataParallel(
                ner_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )
            pretrained_model = torch.nn.parallel.DistributedDataParallel(
                pretrained_model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    logger.info("Try resume from checkpoint")

    if args.restore:
        if os.path.exists(os.path.join(args.output_dir, "global_step.bin")):
            logger.info("Load last checkpoint data")
            global_step = torch.load(os.path.join(args.output_dir, "global_step.bin"))
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            logger.info("Load from output_dir {}".format(output_dir))

            # args = torch.load(os.path.join(output_dir, 'training_args.bin'))
            if hasattr(ner_model, "module"):
                ner_model.module.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))
            else:  # Take care of distributed/parallel training
                ner_model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))

            if hasattr(pretrained_model, "module"):
                pretrained_model.module.load_state_dict(
                    torch.load(os.path.join(output_dir, "pytorch_pretrained_model.bin"))
                )
            else:  # Take care of distributed/parallel training
                pretrained_model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_pretrained_model.bin")))

            global_step += 1
            start_epoch = int(global_step / len(train_dataloader))
            start_step = global_step - start_epoch * len(train_dataloader) - 1
            logger.info("Start from global_step={} epoch={} step={}".format(global_step, start_epoch, start_step))

            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)

        else:
            global_step = 0
            start_epoch = 0
            start_step = 0
            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)

            logger.info("Start from scratch")

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    pretrained_model.zero_grad()
    ner_model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            if args.freeze_bert:
                pretrained_model.eval()
            else:
                pretrained_model.train()
            ner_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "word_ids": batch[3],
            }

            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = ner_model(pretrained_model_outputs, input_ids=inputs["input_ids"], labels=inputs["labels"])
            # ner_model(pretrained_model_outputs, **inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_iterator.set_description("loss {}".format(loss))

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(ner_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                # model.zero_grad()
                pretrained_model.zero_grad()
                ner_model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(args, global_step, ner_model, pretrained_model)

                if (
                    args.local_rank == -1 and args.evaluate_during_training and global_step % args.eval_steps == 0
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(args, model, tokenizer)

                    # Save model if it has improved
                    if prev_eval_loss > results["loss"]:
                        prev_eval_loss = results["loss"]
                        save_model(args, "best-model", ner_model, pretrained_model)

                    # Add to writer
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                logger.info("***** evaluating *****")
                model = (pretrained_model, ner_model)

                results = evaluate(args, model, tokenizer, prefix="")

                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            logger.info("***** evaluating *****")
            model = (pretrained_model, ner_model)

            results = evaluate(args, model, tokenizer, prefix="")

            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            break
        # model = (pretrained_model, ner_model)
        # logger.info("***** evaluating *****")

        # results = evaluate(args, model, tokenizer, prefix="")
        # for key, value in results.items():
        #     tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    #
    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


save_results = []


def evaluate(args, model, tokenizer, dataset_type: str = "dev", prefix=""):
    pretrained_model = model[0]
    ner_model = model[1]

    label_list = ["B-ONT", "B-BUS", "B-RES", "I-ONT", "I-BUS", "I-RES", "O"]
    # label_map = {label: i for i, label in enumerate(label_list)}
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)
    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, dataset_type, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        all_preds = []
        all_targets = []
        out_label_ids = None
        eval_acc = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                pretrained_model.eval()
                ner_model.eval()
                # model.eval()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2],
                    "word_ids": batch[3],
                }
                # outputs = model(**inputs)
                pretrained_model_outputs = pretrained_model(**inputs)
                outputs = ner_model(pretrained_model_outputs, **inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                # logits has shape (bs, seq_len, num_labels)
                logits_batch = logits
                word_ids_batch = inputs["word_ids"]
                targets_batch = inputs["labels"]
                for logits, word_ids, targets in zip(logits_batch, word_ids_batch, targets_batch):
                    mask = word_ids != -1
                    logits = logits[mask]
                    targets = targets[mask]
                    # now i have vectors of various length
                    preds = torch.argmax(logits, dim=-1)

                    preds = preds.tolist()
                    targets = targets.tolist()

                    all_preds.extend(preds)
                    all_targets.extend(targets)

                eval_loss = eval_loss / nb_eval_steps
                accuracy = accuracy_score(all_targets, all_preds)
                precision_scores = precision_score(all_targets, all_preds, average=None)
                recall_scores = recall_score(all_targets, all_preds, average=None)
                f1_scores = f1_score(all_targets, all_preds, average=None)

                # logger.info("Label map:{}".format(label_map))
                if args.logging_steps > 0 and nb_eval_steps % args.logging_steps == 0:
                    logger.info("{} micro f1 scores:{}".format(dataset_type, f1_scores))
                    logger.info("{} recall scores :{}".format(dataset_type, recall_scores))
                    logger.info("{} precision scores:{}".format(dataset_type, precision_scores))
                    logger.info("{} accuracy:{}".format(dataset_type, accuracy))

                result = {"f1": f1_scores, "prec": precision_scores, "recall": recall_scores, "acc": accuracy}

                results[dataset_type] = result
                save_result = str(results)

                save_results.append(save_result)
                result_file = open(os.path.join(args.output_dir, args.my_model_name + "_result.txt"), "w")
                for line in save_results:
                    result_file.write(str(dataset_type) + ":" + str(line) + "\n")
                result_file.close()
    return results


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            dataset_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = (
            processor.get_dev_examples(args.data_dir, dataset_type)
            if evaluate
            else processor.get_train_examples(args.data_dir, dataset_type)
        )
        features = convert_examples_to_features_ner(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),  # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(
                args.model_type in ["roberta"]
            ),  # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_word_ids)
    return dataset


from pytorch_transformers.modeling_bert import BertEncoder


class Adapter(nn.Module):
    def __init__(self, args, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)
        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected, attention_mask=extended_attention_mask, head_mask=head_mask)
        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-self.adapter_config.initializer_range, self.adapter_config.initializer_range)


class PretrainedModel(nn.Module):
    def __init__(self, args):
        super(PretrainedModel, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large", output_hidden_states=True)
        self.config = self.model.config
        self.config.freeze_adapter = args.freeze_adapter
        if args.freeze_bert:
            for p in self.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        word_ids=None,
        labels=None,
        start_id=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_pretrained_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float = 0.1
            hidden_dropout_prob: float = 0.1
            hidden_size: int = 768
            initializer_range: float = 0.02
            intermediate_size: int = 3072
            layer_norm_eps: float = 1e-05
            max_position_embeddings: int = 514
            num_attention_heads: int = 12
            num_hidden_layers: int = self.args.adapter_transformer_layers
            num_labels: int = 2
            output_attentions: bool = False
            output_hidden_states: bool = False
            torchscript: bool = False
            type_vocab_size: int = 1
            vocab_size: int = 50265

        self.adapter_config = AdapterConfig
        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.adapter_list = args.adapter_list
        self.adapter_num = len(self.adapter_list)
        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

    def forward(self, pretrained_model_outputs):
        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to("cuda")
        # hidden_states_last = torch.zeros(sequence_output.size()).to("cpu")

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = (
                        hidden_states_last
                        + adapter_hidden_states[int(adapter_hidden_states_count / self.adapter_skip_layers)]
                    )

        outputs = (hidden_states_last,) + outputs[2:]

        return outputs  # (loss), logits, (hidden_states), (attentions)


class NERModel(nn.Module):
    def __init__(self, args, pretrained_model_config, fac_adapter, et_adapter, lin_adapter):
        super(NERModel, self).__init__()
        self.args = args
        self.config = pretrained_model_config

        # self.adapter = AdapterModel(self.args, pretrained_model_config)
        self.fac_adapter = fac_adapter
        self.ner_adapter = et_adapter
        self.lin_adapter = lin_adapter
        if args.freeze_adapter and (self.fac_adapter is not None):
            for p in self.fac_adapter.parameters():
                p.requires_grad = False
        # if args.freeze_adapter and (self.ner_adapter is not None):
        #     for p in self.ner_adapter.parameters():
        #         p.requires_grad = False
        # if args.freeze_adapter and (self.lin_adapter is not None):
        #     for p in self.lin_adapter.parameters():
        #         p.requires_grad = False
        self.adapter_num = 0
        if self.fac_adapter is not None:
            self.adapter_num += 1
        # if self.ner_adapter is not None:
        #     self.adapter_num += 1
        # if self.lin_adapter is not None:
        #     self.adapter_num += 1

        if self.args.fusion_mode == "concat":
            self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

        # self.num_labels = config.num_labels
        # self.num_labels = 9
        self.num_labels = args.num_labels
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, self.num_labels)

    @staticmethod
    def loss_fn(logits, target):
        lfn = torch.nn.CrossEntropyLoss()

        # Filter representative tokens
        active_logits = logits.view(-1, logits.shape[-1])
        active_target = target.view(-1)

        # Calculate loss
        loss = lfn(active_logits, active_target)

        return loss

    def forward(
        self,
        pretrained_model_outputs,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        word_ids=None,
        labels=None,
        start_id=None,
    ):
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]
        # if self.fac_adapter is not None:
        fac_adapter_outputs, _ = self.fac_adapter(pretrained_model_outputs)
        # if self.ner_adapter is not None:
        #     ner_adapter_outputs, _ = self.ner_adapter(pretrained_model_outputs)
        # if self.lin_adapter is not None:
        #     lin_adapter_outputs, _ = self.lin_adapter(pretrained_model_outputs)
        if self.args.fusion_mode == "add":
            task_features = pretrained_model_last_hidden_states
            if self.fac_adapter is not None:
                task_features = task_features + fac_adapter_outputs
            # if self.ner_adapter is not None:
            #     task_features = task_features + ner_adapter_outputs
            # if self.lin_adapter is not None:
            #     task_features = task_features + lin_adapter_outputs
        elif self.args.fusion_mode == "concat":
            combine_features = pretrained_model_last_hidden_states
            # if self.fac_adapter is not None and self.lin_adapter is not None:
            #     fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
            #     lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
            #     task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))
            # elif self.fac_adapter is not None:
            task_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))

        logits = self.out_proj(self.dropout(self.dense(task_features)))

        outputs = (logits,) + pretrained_model_outputs[2:]
        if labels is not None:

            # loss_fct = CrossEntropyLoss()
            loss = self.loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


def save_model(args, global_step, ner_model, pretrained_model):
    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (
        ner_model.module if hasattr(ner_model, "module") else ner_model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    model_to_save = (
        pretrained_model.module if hasattr(pretrained_model, "module") else pretrained_model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))

    torch.save(global_step, os.path.join(args.output_dir, "global_step.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)


def load_pretrained_adapter(adapter, adapter_path):
    new_adapter = adapter
    model_dict = new_adapter.state_dict()
    adapter_meta_dict = torch.load(adapter_path, map_location=lambda storage, loc: storage)
    for item in [
        "out_proj.bias",
        "out_proj.weight",
        "dense.weight",
        "dense.bias",
    ]:  # 'adapter.down_project.weight','adapter.down_project.bias','adapter.up_project.weight','adapter.up_project.bias'
        if item in adapter_meta_dict:
            adapter_meta_dict.pop(item)

    changed_adapter_meta = {}
    for key in adapter_meta_dict.keys():
        changed_adapter_meta[key.replace("adapter.", "adapter.")] = adapter_meta_dict[key]

    changed_adapter_meta = {k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()}
    model_dict.update(changed_adapter_meta)
    new_adapter.load_state_dict(model_dict)
    return new_adapter


class FinetuneKAdapterArgs(object):
    def __init__(self) -> None:

        self.model_type = "roberta"
        self.model_name = "roberta-large"
        self.model_name_or_path = "roberta-large"
        self.data_dir = "./data/ner_data/finetuning"
        self.output_dir = "ner_output"
        self.restore = True
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = True
        self.task_name = "ner"
        self.comment = "fac-adapter"
        self.per_gpu_train_batch_size = 32
        self.per_gpu_eval_batch_size = 64
        self.num_train_epochs = 4
        self.max_seq_lengt = 64
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.warmup_steps = 100
        self.save_steps = 1e8
        self.eval_steps = 1000
        self.adapter_size = 768
        self.adapter_list = "0,11,22"
        self.adapter_skip_layers = 0
        self.adapter_transformer_layers = 2
        self.meta_adapter_model = ""
        self.max_seq_length = 256
        self.no_cuda = False
        self.fusion_mode = "concat"  # "add"
        self.meta_fac_adaptermodel = "output_data/custom_maxlen-256_batch-32_lr-5e-05_warmup-1200_epoch-8_fac-adapter/checkpoint-best-model/pytorch_model.bin"


def main(special_args=None):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument("--comment", default="", type=str, help="The comment")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--freeze_bert", default=True, type=bool, help="freeze the parameters of pretrained model.")
    parser.add_argument("--freeze_adapter", default=False, type=bool, help="freeze the parameters of adapter.")

    parser.add_argument("--test_mode", default=0, type=int, help="test freeze adapter")

    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="concat",
        help="the fusion mode for bert feautre and adapter feature |add|concat",
    )
    parser.add_argument("--adapter_transformer_layers", default=2, type=int, help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=128, type=int, help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str, help="The layer where add an adapter")
    parser.add_argument(
        "--adapter_skip_layers", default=6, type=int, help="The skip_layers of adapter according to bert layers"
    )

    parser.add_argument("--meta_fac_adaptermodel", default="", type=str, help="the pretrained factual adapter model")
    parser.add_argument(
        "--meta_et_adaptermodel", default="", type=str, help="the pretrained entity typing adapter model"
    )
    parser.add_argument("--meta_lin_adaptermodel", default="", type=str, help="the pretrained linguistic adapter model")

    ## Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3"
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=None, help="eval every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--restore",
        type=bool,
        default=True,
        help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--meta_bertmodel", default="", type=str, help="the pretrained bert model")
    parser.add_argument("--save_model_iteration", type=int, help="when to save the model..")
    args = parser.parse_args()

    if special_args is not None:
        args.model_type = special_args.model_type
        args.model_name = special_args.model_name
        args.model_name_or_path = special_args.model_name_or_path
        args.data_dir = special_args.data_dir
        args.output_dir = special_args.output_dir
        args.restore = special_args.restore
        args.do_train = special_args.do_train
        args.do_eval = special_args.do_eval
        args.evaluate_during_training = special_args.evaluate_during_training
        args.task_name = special_args.task_name
        args.comment = special_args.comment
        args.per_gpu_train_batch_size = special_args.per_gpu_train_batch_size
        args.per_gpu_eval_batch_size = special_args.per_gpu_eval_batch_size
        args.num_train_epochs = special_args.num_train_epochs
        args.max_seq_lengt = special_args.max_seq_lengt
        args.gradient_accumulation_steps = special_args.gradient_accumulation_steps
        args.learning_rate = special_args.learning_rate
        args.warmup_steps = special_args.warmup_steps
        args.save_steps = special_args.save_steps
        args.eval_steps = special_args.eval_steps
        args.adapter_size = special_args.adapter_size
        args.adapter_list = special_args.adapter_list
        args.adapter_skip_layers = special_args.adapter_skip_layers
        args.adapter_transformer_layers = special_args.adapter_transformer_layers
        args.meta_adapter_model = special_args.meta_adapter_model
        args.max_seq_length = special_args.max_seq_length
        args.no_cuda = special_args.no_cuda
        args.fusion_mode = special_args.fusion_mode
        args.meta_fac_adaptermodel = special_args.meta_fac_adaptermodel

    args.adapter_list = args.adapter_list.split(",")
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = (
        "batch-"
        + str(args.per_gpu_train_batch_size)
        + "_"
        + "lr-"
        + str(args.learning_rate)
        + "_"
        + "warmup-"
        + str(args.warmup_steps)
        + "_"
        + "epoch-"
        + str(args.num_train_epochs)
        + "_"
        + str(args.comment)
    )
    args.my_model_name = args.task_name + "_" + name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large", add_prefix_space=True)
    pretrained_model = PretrainedModel(args)
    if args.meta_fac_adaptermodel:
        fac_adapter = AdapterModel(args, pretrained_model.config)
        fac_adapter = load_pretrained_adapter(fac_adapter, args.meta_fac_adaptermodel)
    else:
        fac_adapter = None
    if args.meta_et_adaptermodel:
        ner_adapter = AdapterModel(args, pretrained_model.config)
        ner_adapter = load_pretrained_adapter(ner_adapter, args.meta_et_adaptermodel)
    else:
        ner_adapter = None
    if args.meta_lin_adaptermodel:
        lin_adapter = AdapterModel(args, pretrained_model.config)
        lin_adapter = load_pretrained_adapter(lin_adapter, args.meta_lin_adaptermodel)
    else:
        lin_adapter = None
    # adapter_model = AdapterModel(pretrained_model.config,num_labels,args.adapter_size,args.adapter_interval,args.adapter_skip_layers)
    ner_model = NERModel(
        args, pretrained_model.config, fac_adapter=fac_adapter, et_adapter=ner_adapter, lin_adapter=lin_adapter
    )

    if args.meta_bertmodel:
        model_dict = pretrained_model.state_dict()
        bert_meta_dict = torch.load(args.meta_bertmodel, map_location=lambda storage, loc: storage)
        for item in [
            "out_proj.weight",
            "out_proj.bias",
            "dense.weight",
            "dense.bias",
            "lm_head.bias",
            "lm_head.dense.weight",
            "lm_head.dense.bias",
            "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias",
            "lm_head.decoder.weight",
        ]:
            if item in bert_meta_dict:
                bert_meta_dict.pop(item)

        changed_bert_meta = {}
        for key in bert_meta_dict.keys():
            changed_bert_meta[key.replace("model.", "roberta.")] = bert_meta_dict[key]
        # print(changed_bert_meta.keys())
        changed_bert_meta = {k: v for k, v in changed_bert_meta.items() if k in model_dict.keys()}
        # print(changed_bert_meta.keys())
        model_dict.update(changed_bert_meta)
        pretrained_model.load_state_dict(model_dict)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    pretrained_model.to(args.device)
    ner_model.to(args.device)
    # model.to(args.device)
    model = (pretrained_model, ner_model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, "train", evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            ner_model.module if hasattr(ner_model, "module") else ner_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = os.path.join(args.output_dir, "checkpoint-best-model")
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, dataset_type="test", prefix="best-model")
        logger.info("micro f1:{}".format(result))
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)
    save_result = str(results)
    save_results.append(save_result)

    result_file = open(os.path.join(args.output_dir, args.my_model_name + "_result.txt"), "w")
    for line in save_results:
        result_file.write(str(line) + "\n")
    result_file.close()

    # return results


if __name__ == "__main__":
    arg = FinetuneKAdapterArgs()
    main(special_args=arg)
