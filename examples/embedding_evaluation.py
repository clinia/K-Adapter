import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from run_finetune_openentity_adapter import AdapterModel, load_pretrained_adapter
from search_utils.embeddings_eval.base import EvaluateEmbeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import RobertaTokenizerFast

from pytorch_transformers.modeling_roberta import RobertaModel


class EvalKBERTEMbeddings(EvaluateEmbeddings):
    def _forward(self, model, inputs, labels, tokenizer, device):

        pretrained_model = PretrainedModel().to(device)
        et_model = model

        ent_emb = []

        for i in range(2):
            out = tokenizer(
                list(inputs[i]),
                add_special_tokens=True,
                max_length=self.seq_len,
                padding=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )

            input_ids = torch.LongTensor(out["input_ids"]).to(device)
            attention_mask = torch.LongTensor(out["attention_mask"]).to(device)
            token_type_ids = torch.LongTensor(out["token_type_ids"]).to(device)
            token_type_ids = torch.ones_like(token_type_ids).to(
                device
            )  # They use ones as their ids for single sequences

            # create pos embedding according to how the authors of K-BERT did it

            # forward pass to the model
            # output = model(input_ids, attention_mask, token_type_ids, pos)
            with torch.no_grad():
                inputs_model = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            pretrained_model_output = pretrained_model(**inputs_model)
            output_features = et_model(pretrained_model_output, **inputs_model)

            # Separate embeddings
            # Remove special token indexes from token_tyoe_ids
            token_type_ids = torch.where(((input_ids == 0) | (input_ids == 2) | (input_ids == 1)), 0, token_type_ids)

            # Gather and process embeddings that do not contain special tokens
            emb1 = output_features[token_type_ids == 1, :]
            n_sub_tokens = torch.sum(token_type_ids, dim=1)  # Nb of tokens to sum together for each entity

            ent_emb_i = []
            for n in n_sub_tokens:
                tmp = emb1[: int(n)].sum(0)  # get tokens
                emb1 = emb1[int(n) :]  # remove from tensor to process sequentially the entities in the batch
                ent_emb_i.append(tmp)

            ent_emb.append(torch.stack(ent_emb_i))

        return ent_emb[0], ent_emb[1]
        # return new_emb_0, new_emb_1


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main(DATA_PATH, EXPORT_PATH):

    ####################################
    # Same soup to call the model

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument("--pretrained_model_path")
    parser.add_argument("--comment", default="", type=str, help="The comment")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
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
    parser.add_argument("--adapter_size", default=768, type=int, help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,22", type=str, help="The layer where add an adapter")
    parser.add_argument(
        "--adapter_skip_layers", default=3, type=int, help="The skip_layers of adapter according to bert layers"
    )

    parser.add_argument("--meta_fac_adaptermodel", default="", type=str, help="the pretrained factual adapter model")
    parser.add_argument(
        "--meta_et_adaptermodel", default="", type=str, help="the pretrained entity typing adapter model"
    )
    parser.add_argument("--meta_lin_adaptermodel", default="", type=str, help="the pretrained linguistic adapter model")

    # Other parameters
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
        default=128,
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

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
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

    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
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

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--meta_bertmodel", default="", type=str, help="the pretrained bert model")
    parser.add_argument("--save_model_iteration", type=int, help="when to save the model..")
    args = parser.parse_args()

    args.model_name_or_path = "roberta-large"
    args.model_name = "roberta-large"

    args.adapter_list = args.adapter_list.split(",")
    args.adapter_list = [int(i) for i in args.adapter_list]

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

    # Set seed
    # set_seed(args)
    pretrained_model = PretrainedModel()
    if args.meta_fac_adaptermodel:
        fac_adapter = AdapterModel(args, pretrained_model.config)
        fac_adapter = load_pretrained_adapter(fac_adapter, args.meta_fac_adaptermodel)
    else:
        fac_adapter = None
    if args.meta_et_adaptermodel:
        et_adapter = AdapterModel(args, pretrained_model.config)
        et_adapter = load_pretrained_adapter(et_adapter, args.meta_et_adaptermodel)
    else:
        et_adapter = None
    if args.meta_lin_adaptermodel:
        lin_adapter = AdapterModel(args, pretrained_model.config)
        lin_adapter = load_pretrained_adapter(lin_adapter, args.meta_lin_adaptermodel)
    else:
        lin_adapter = None
    # adapter_model = AdapterModel(pretrained_model.config,num_labels,args.adapter_size,args.adapter_interval,args.adapter_skip_layers)
    model = ETModel(
        args, pretrained_model.config, fac_adapter=fac_adapter, et_adapter=et_adapter, lin_adapter=lin_adapter
    )

    model_dict = model.state_dict()

    model_meta_dict = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
    for item in [
        "out_proj.bias",
        "out_proj.weight",
        "dense.weight",
        "dense.bias",
    ]:  # 'adapter.down_project.weight','adapter.down_project.bias','adapter.up_project.weight','adapter.up_project.bias'
        if item in model_meta_dict:
            model_meta_dict.pop(item)

    model_dict.update(model_meta_dict)

    model.load_state_dict(model_dict)

    ##########################
    # Feed loaded model and tokenizer to the embedding evaluation module

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name_or_path)

    evaluator = EvalKBERTEMbeddings(DATA_PATH=DATA_PATH, batch_size=10, seq_len=30)
    results = evaluator.evaluate(model, tokenizer)

    evaluator.export(results=results, EXPORT_PATH=EXPORT_PATH)


class PretrainedModel(nn.Module):
    def __init__(self):
        super(PretrainedModel, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large", output_hidden_states=True)
        self.config = self.model.config

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        return outputs  # (loss), logits, (hidden_states), (attentions)


class ETModel(nn.Module):
    def __init__(self, args, pretrained_model_config, fac_adapter, et_adapter, lin_adapter):
        super(ETModel, self).__init__()
        self.args = args
        self.config = pretrained_model_config

        # self.adapter = AdapterModel(self.args, pretrained_model_config)
        self.fac_adapter = fac_adapter
        self.et_adapter = et_adapter
        self.lin_adapter = lin_adapter
        if args.freeze_adapter and (self.fac_adapter is not None):
            for p in self.fac_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.et_adapter is not None):
            for p in self.et_adapter.parameters():
                p.requires_grad = False
        if args.freeze_adapter and (self.lin_adapter is not None):
            for p in self.lin_adapter.parameters():
                p.requires_grad = False
        self.adapter_num = 0
        if self.fac_adapter is not None:
            self.adapter_num += 1
        if self.et_adapter is not None:
            self.adapter_num += 1
        if self.lin_adapter is not None:
            self.adapter_num += 1

        if self.args.fusion_mode == "concat":
            self.task_dense_lin = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense_fac = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)
            self.task_dense = nn.Linear(self.config.hidden_size + self.config.hidden_size, self.config.hidden_size)

    def forward(
        self,
        pretrained_model_outputs,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        start_id=None,
    ):
        pretrained_model_last_hidden_states = pretrained_model_outputs[0]
        if self.fac_adapter is not None:
            fac_adapter_outputs, _ = self.fac_adapter(pretrained_model_outputs)
        if self.et_adapter is not None:
            et_adapter_outputs, _ = self.et_adapter(pretrained_model_outputs)
        if self.lin_adapter is not None:
            lin_adapter_outputs, _ = self.lin_adapter(pretrained_model_outputs)
        if self.args.fusion_mode == "add":
            task_features = pretrained_model_last_hidden_states
            if self.fac_adapter is not None:
                task_features = task_features + fac_adapter_outputs
            if self.et_adapter is not None:
                task_features = task_features + et_adapter_outputs
            if self.lin_adapter is not None:
                task_features = task_features + lin_adapter_outputs
        elif self.args.fusion_mode == "concat":
            combine_features = pretrained_model_last_hidden_states
            fac_features = self.task_dense_fac(torch.cat([combine_features, fac_adapter_outputs], dim=2))
            lin_features = self.task_dense_lin(torch.cat([combine_features, lin_adapter_outputs], dim=2))
            task_features = self.task_dense(torch.cat([fac_features, lin_features], dim=2))

        return task_features  # (loss), logits, (hidden_states), (attentions)


if __name__ == "__main__":
    DATA_PATH = "./data/data.csv"
    EXPORT_PATH = "./outputs/embedding_eval"

    main(DATA_PATH, EXPORT_PATH)
