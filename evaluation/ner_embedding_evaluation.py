import os
import sys
import logging
import argparse
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from evaluation.models import PretrainedModel, NERModel, AdapterModel, load_pretrained_adapter
from examples.utils_glue import convert_examples_to_features_ner, output_modes, processors

from search_utils.embeddings_eval.base import EvaluateEmbeddings


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class EvalNEREmbeddings(EvaluateEmbeddings):

    """
    NOTE: this is single-usage script, it is not currently configured in any pipeline or stage.
    """

    def evaluate(self, model, tokenizer, no_cuda: bool = False, mode: str = "sum") -> pd.DataFrame:

        # send model to device
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        if torch.cuda.device_count() > 1:
            print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

        pretrained_model, ner_model = model
        pretrained_model = pretrained_model.to(device)
        ner_model = ner_model.to(device)

        pretrained_model.eval()
        ner_model.eval()
        model = (pretrained_model, ner_model)
        # Start evaluation loop
        with torch.no_grad():

            pos_couple_means = []
            neg_couple_means = []
            sim_vector = []
            for i, (
                inputs,
                labels,
            ) in enumerate(self.dataloader):

                embd1, embd2 = self._forward(model, inputs, labels, tokenizer, device, mode)

                # Handle case where batch size is one
                if len(embd1.shape) != 2:
                    embd1 = embd1.unsqueeze(0)
                    embd2 = embd2.unsqueeze(0)

                similarity = self.cos(embd1, embd2)

                sim_vector.extend(list(similarity.detach().cpu().numpy()))

                positive = similarity[labels == 1]
                negative = 1 - similarity[labels == 0]

                batch_pos_mean = torch.mean(positive)
                batch_neg_mean = torch.mean(negative)

                pos_couple_means.append(batch_pos_mean.detach().cpu().numpy())
                neg_couple_means.append(batch_neg_mean.detach().cpu().numpy())

            pos_mean = sum(pos_couple_means) / len(pos_couple_means)
            neg_mean = sum(neg_couple_means) / len(neg_couple_means)

            print("Mean similarity for positive examples : ", pos_mean)
            print("Mean inverse similarity for negative examples : ", neg_mean)
            print("Total mean similarity is : ", (pos_mean + neg_mean) / 2)

        self.dataset.dataset["similarity_score"] = pd.Series(sim_vector)
        return self.dataset.dataset

    def _forward(self, model, inputs, labels, tokenizer, device, mode):

        pretrained_model = model[0]
        ner_model = model[1]

        ent_emb = []
        for i in range(2):

            tokenization = tokenizer(
                text=list(inputs[i]),
                add_special_tokens=True,
                max_length=self.seq_len,
                truncation="longest_first",
                padding="max_length",
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            ## Separate embeddings
            # Remove special token indexes from token_tyoe_ids

            input_ids = tokenization["input_ids"].to(device)
            attention_mask = tokenization["attention_mask"].to(device)
            token_type_ids = tokenization["token_type_ids"].to(device)
            token_type_ids = torch.ones_like(token_type_ids).to(
                device
            )  # They use ones as their ids for single sequences
            token_type_ids = torch.where(
                (
                    (input_ids == tokenizer.bos_token_id)
                    | (input_ids == tokenizer.pad_token_id)
                    | (input_ids == tokenizer.eos_token_id)
                ),
                0,
                token_type_ids,
            )

            pretrained_model_outputs = pretrained_model(input_ids, attention_mask=attention_mask)
            outputs = ner_model(pretrained_model_outputs, input_ids=input_ids, labels=None)

            if mode == "sum":
                token_embeddings = outputs[1]
            if mode == "concat":
                token_embeddings = outputs[2]
            emb1 = token_embeddings[token_type_ids == 1, :]
            n_sub_tokens = torch.sum(token_type_ids, dim=1)  # Nb of tokens to sum together for each entity

            ent_emb_i = []
            for n in n_sub_tokens:
                tmp = emb1[: int(n)].sum(0)  # get tokens
                emb1 = emb1[int(n) :]  # remove from tensor to process sequentially the entities in the batch
                ent_emb_i.append(tmp)

            ent_emb.append(torch.stack(ent_emb_i))

        return ent_emb[0], ent_emb[1]


class FinetuneKAdapterArgs(object):
    def __init__(self) -> None:

        self.model_type = "roberta"
        self.model_name = "roberta-base"
        self.model_name_or_path = "roberta-base"
        self.data_dir = "./data/ner_data/finetuning"
        self.output_dir = "ner_output"
        self.restore = False
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = True
        self.task_name = "ner"
        self.comment = "fac-adapter"
        self.per_gpu_train_batch_size = 50
        self.per_gpu_eval_batch_size = 128
        self.num_train_epochs = 4
        self.max_seq_lengt = 64
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.warmup_steps = 1000
        self.save_steps = 1e8
        self.eval_steps = 1000
        self.adapter_size = 768
        self.adapter_list = "0,11,22"
        self.adapter_skip_layers = 0
        self.adapter_transformer_layers = 2
        self.meta_adapter_model = ""
        self.max_seq_length = 256
        self.no_cuda = True
        self.fusion_mode = "concat"  # "add"
        self.meta_fac_adaptermodel = "ner_output/ner_batch-600_lr-0.0005_warmup-50_epoch-4_fac-adapter-plus/checkpoint-best-model/pytorch_model.bin"
        self.embd_type = "concat"


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
        help="Model type selected",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut",
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
        args.embd_type = special_args.embd_type

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

    evaluator = EvalNEREmbeddings(DATA_PATH="evaluation/data/data.csv", batch_size=10, seq_len=64)

    results = evaluator.evaluate(model, tokenizer=tokenizer, no_cuda=args.no_cuda, mode=args.embd_type)

    EXPORT_PATH = ""
    evaluator.export(results, EXPORT_PATH)


if __name__ == "__main__":
    arg = FinetuneKAdapterArgs()
    main(special_args=arg)
