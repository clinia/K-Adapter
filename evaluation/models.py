import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertEncoder
from pytorch_transformers import WEIGHTS_NAME, RobertaModel


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
        return up_projected

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
        self.model = RobertaModel.from_pretrained(args.model_name, output_hidden_states=True)
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
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)
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

        ### Another Bert Layer
        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = 768
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
        self.encoder = BertEncoder(self.adapter_config)

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
                concat_features = task_features
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
            concat_features = torch.cat([combine_features, fac_adapter_outputs], dim=2)
            task_features = self.task_dense_fac(concat_features)

        # Pass to BERT layer
        input_shape = task_features.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # head_mask = [None] * self.adapter_config.num_hidden_layers
        # encoder_outputs = self.encoder(task_features, attention_mask=extended_attention_mask, head_mask=head_mask)

        logits = self.out_proj(self.dropout(self.dense(task_features)))

        outputs = (logits,) + pretrained_model_outputs[2:]
        if labels is not None:

            # loss_fct = CrossEntropyLoss()
            loss = self.loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs, task_features, fac_adapter_outputs  # (loss), logits, (hidden_states), (attentions) = outputs


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
