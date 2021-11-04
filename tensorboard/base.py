import sys
import json
import logging
import os
import time
import argparse
import random
from io import DEFAULT_BUFFER_SIZE
from typing import Dict, Generator, List, Union

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast
import tensorboard as tb
import yaml
from genericpath import exists
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from evaluation.models import PretrainedModel, NERModel, AdapterModel, load_pretrained_adapter
from examples.utils_glue import convert_examples_to_features_ner, output_modes, processors

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class VisualizeEmbeddings:
    """
    A class that allows 3D visualization of embeddings via Tensorboard Projector. Visualization can be accomplished in two ways. First by passing a pre-computed dictionary of embeddings in the following format
    {id:{text: str(), embedding:list()}} to the create_checkpoint method followed by the visualize method. The second option makes use of the import_dict method, which automatically checks for a cached version
    of the desired embeddings or computes them if they don't exist. The second option should be called via the main() method.
    """

    def __init__(
        self,
        model,
        tokenizer,
        seq_len,
        device,
        embedding_src,
        LOG_DIR_BASE: str,
        DEFAULT_BUS_PATH: str,
        DEFAULT_ONT_PATH: str,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.embedding_src = embedding_src
        self.LOG_DIR_BASE = LOG_DIR_BASE
        self.DEFAULT_BUS_PATH = DEFAULT_BUS_PATH
        self.DEFAULT_ONT_PATH = DEFAULT_ONT_PATH
        self.batch_size = batch_size
        self.device = device

        # Abbreviation dict
        self.facets = {
            "ser": "service",
            "prd": "product",
            "prf": "profession",
            "spc": "specialty",
            "res": "resource",
            "all": "all",
        }

        # Load logger
        self.logger = logging.getLogger(__name__)

    def view(
        self,
        label: Union[str, List[str]],
        model: str,
        mode: str,
        lang: str,
        force_recompute: bool = False,
        launch_server: bool = True,
    ) -> None:
        """
        Main logic to execute when creating visualizations in Tensorboard. This implementation supports the creation of views
        containing multiple sources of embeddings (ie. ONT + BUS).
        """
        if mode not in ("cls", "sum", "mean"):
            raise ValueError("mode parameter must be 'sum' or 'mode'.")

        log_dir = "{}/{}/{}".format(self.LOG_DIR_BASE, lang, model)
        name = "{}_{}_embedding".format(mode, "_".join(label) if isinstance(label, list) else label)

        label = label if isinstance(label, list) else [label]
        label.sort()  # Ensure same color for labels and  cache filename

        # Collect embeddings
        embeddings_dict_list = []
        for l in label:
            embeddings_dict_list.append(self.import_dict(log_dir, l.lower(), model, mode, lang, force=force_recompute))

        # Add labels if multiple dicts
        for d, l in zip(embeddings_dict_list, label):
            for key in d:
                d[key]["label"] = l.upper()

        # Join dictionaries
        embeddings_dict = {}
        for d in embeddings_dict_list:
            embeddings_dict = {**embeddings_dict, **d}

        self.create_checkpoint(embedding_dict=embeddings_dict, log_dir=log_dir, name=name)

        if launch_server:
            self.serve(log_dir)

    def serve(self, tracking_address) -> None:
        """Launches the Tensorboard server endpoint at http://localhost:6006/"""
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", tracking_address])
        url = tb.launch()
        while True:
            time.sleep(5)

    def create_checkpoint(self, embedding_dict: dict, log_dir: str, name: str) -> None:
        """Extracts informations form the input dictionary and creates a checkpoint readable by Tensorboard"""

        # Extract embeddings and tags (label and title)
        text = list(embedding_dict.keys())
        values = list(embedding_dict.values())
        vectors = np.array([item["embedding"] for item in values])
        labels = [item["label"] for item in values]
        metadata = list(zip(text, labels))

        # Create a Tensorboard event
        writer = SummaryWriter(log_dir="{}/{}".format(log_dir, name))
        writer.add_embedding(vectors, metadata, metadata_header=["title", "label"])
        writer.close()

    def import_dict(self, log_dir: str, label: str, model: str, mode: str, lang: str, force: bool) -> Dict:
        """Imports the desired embedding dict or computes the embedding by calling the appropriate model"""

        root_path = "{}/embeddings".format(log_dir)
        path = "{}/{}_{}_embedding_cache.json".format(root_path, mode, label)

        if os.path.isfile(path) and not force:

            self.logger.info("Loading embeddings from cache...")
            with open(path, "r") as f:
                embeddings = json.load(f)

            # Filtering duplicates
            embedding_dict = dict()
            for id, info in embeddings.items():
                if info not in embedding_dict.values():
                    embedding_dict[id] = info
        else:
            t1 = time.time()
            self.logger.info("No cache found! Computing embeddings...")
            embedding_dict = self.compute_embeddings(label, mode=mode, lang=lang)

            self.logger.info("Computation took: {}".format(time.time() - t1))

            os.makedirs(root_path, exist_ok=True)

            # Save computed embeddings
            with open(path, "w") as f:
                json.dump(embedding_dict, f)

        return embedding_dict

    def compute_embeddings(self, label: str, mode: str, lang: str) -> Dict:
        """Creates the embeddings by calling the model and processing the outputs according to the selected mode"""
        mean_embeddings = True if mode == "mean" else False

        ## Load and prepare data
        if label in {"ser", "prd", "prf", "spc", "res", "all"}:
            data = self._import_facet(DEFAULT_ONT_PATH=self.DEFAULT_ONT_PATH, label=label, lang=lang)

        if label == "ont":
            data = []
            keys = list(self.facets.keys())
            keys.remove("res")
            for facet in keys:
                data.extend(self._import_facet(DEFAULT_ONT_PATH=self.DEFAULT_ONT_PATH, label=facet, lang=lang))
        if label == "bus":
            data = self._import_business(PATH="{}/{}/documents.json".format(os.getcwd(), self.DEFAULT_BUS_PATH))

        # send model to device
        pretrained_model, ner_model = self.model
        pretrained_model = pretrained_model.to(self.device)
        ner_model = ner_model.to(self.device)

        pretrained_model.eval()
        ner_model.eval()
        model = (pretrained_model, ner_model)
        # Send to NER
        with torch.no_grad():
            embeddings = list()
            for titles_batch in tqdm(self._batch(data, self.batch_size), total=len(data) // self.batch_size):
                pretrained_model = model[0]
                ner_model = model[1]

                tokenization = self.tokenizer(
                    text=list(titles_batch),
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

                input_ids = tokenization["input_ids"].to(self.device)
                attention_mask = tokenization["attention_mask"].to(self.device)
                token_type_ids = tokenization["token_type_ids"].to(self.device)
                token_type_ids = torch.ones_like(token_type_ids).to(
                    self.device
                )  # They use ones as their ids for single sequences
                token_type_ids = torch.where(
                    (
                        (input_ids == self.tokenizer.bos_token_id)
                        | (input_ids == self.tokenizer.pad_token_id)
                        | (input_ids == self.tokenizer.eos_token_id)
                    ),
                    0,
                    token_type_ids,
                )

                pretrained_model_outputs = pretrained_model(input_ids, attention_mask=attention_mask)
                outputs = ner_model(pretrained_model_outputs, input_ids=input_ids, labels=None)

                if self.embedding_src == "sum":
                    token_embeddings = outputs[1]
                if self.embedding_src == "concat":
                    token_embeddings = outputs[2]
                emb1 = token_embeddings[token_type_ids == 1, :]
                n_sub_tokens = torch.sum(token_type_ids, dim=1)  # Nb of tokens to sum together for each entity

                ent_emb_i = []
                for n in n_sub_tokens:
                    divider = n if mean_embeddings else 1
                    tmp = emb1[: int(n)].sum(0)  # get tokens
                    tmp = tmp / divider
                    emb1 = emb1[int(n) :]  # remove from tensor to process sequentially the entities in the batch
                    ent_emb_i.append(tmp)

                embeddings.extend(torch.stack(ent_emb_i))

        # Process embeddings
        # embeddings = self._get_embeddings(outputs, mode=mode, mean_embeddings=mean_embeddings)

        # Build dict
        embeddings_dict = dict()
        for title, embedding in zip(data, embeddings):
            embeddings_dict[title] = {"label": label.upper(), "embedding": embedding.tolist()}

        return embeddings_dict

    def _import_facet(self, DEFAULT_ONT_PATH: str, label: str, lang: str) -> List:
        """Returns the concepts as a dictionary with the id as the key and the value as the name in the correct language."""
        path = "{}/{}/{}/name.csv".format(os.getcwd(), DEFAULT_ONT_PATH, self.facets[label])

        facet = pd.read_csv(path, names=["name"])
        facet = list(set(facet.name.to_list()))
        return facet

    @staticmethod
    def _import_business(PATH: str) -> List:
        """Returns the businesses as a dictionary with the id as the key and the value as the name in the correct language."""
        with open(PATH, "r") as f:
            business_dict = json.load(f)

        bus_names = [bus["identifiers"]["standard"][0] for bus in business_dict]
        bus_names = list(set(bus_names))
        return bus_names

    @staticmethod
    def _batch(iterable, batch_size=1) -> Generator:
        """Helper tool to batch elements to be passed onto the model"""
        l = len(iterable)
        for ndx in range(0, l, batch_size):
            yield iterable[ndx : min(ndx + batch_size, l)]

    def _get_embeddings(self, responses: List, mode: str, mean_embeddings: bool = False) -> List:
        """Gather the embeddings from the model's response according to the selected mode"""
        if mode == "cls":
            embeddings_list = [response["cls_embedding"] for response in responses]
        elif mode == "sum" or mode == "mean":
            embeddings_list = [
                self._create_sum_embeddings([term["embedding"] for term in response["terms"]], mean_embeddings)
                for response in responses
            ]
        return embeddings_list

    @staticmethod
    def _create_sum_embeddings(embeddings_lst: List, mean_embeddings: bool = False) -> List:
        """Processes a list of embeddings into a sum and the averages if enabled"""
        n = len(embeddings_lst)

        embeddings = np.sum(np.array(embeddings_lst), 0)
        if mean_embeddings and n != 0:
            embeddings = embeddings / n
        return list(embeddings)


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
        self.comment = "fac-mse-last"
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
        self.adapter_list = "11"
        self.adapter_skip_layers = 0
        self.adapter_transformer_layers = 6
        self.meta_adapter_model = ""
        self.max_seq_length = 64
        self.no_cuda = False
        self.fusion_mode = "concat"  # "add"
        self.meta_fac_adaptermodel = "output_data/custom_maxlen-64_batch-200_lr-0.0005_warmup-15_epoch-25_fac-mse-last-reduced-linear/checkpoint-best-model/pytorch_model.bin"  # "ner_output/ner_batch-600_lr-0.0005_warmup-50_epoch-4_fac-adapter-plus/checkpoint-best-model/pytorch_model.bin"
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

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name, add_prefix_space=True)
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

    # Initialize logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOG_DIR_BASE = "tensorboard/results"
    DEFAULT_ONT_PATH = "tensorboard/data/facets"
    DEFAULT_BUS_PATH = "model/input_data/tensorboard"

    visio = VisualizeEmbeddings(
        model,
        tokenizer,
        embedding_src="concat",
        seq_len=args.max_seq_length,
        LOG_DIR_BASE=LOG_DIR_BASE,
        DEFAULT_BUS_PATH=DEFAULT_BUS_PATH,
        DEFAULT_ONT_PATH=DEFAULT_ONT_PATH,
        device=args.device,
    )

    visio.view(label="all", model="roberta-mse-linear", mode="sum", lang="en", force_recompute=True)


if __name__ == "__main__":
    arg = FinetuneKAdapterArgs()
    main(special_args=arg)
