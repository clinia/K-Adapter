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

from __future__ import absolute_import, division, print_function

import copy
import csv
import json
import logging
import os
import sys
from io import open

import numpy as np
import torch
from transformers import RobertaTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, start_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.start_id = start_id


class mlmInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, basic_mask, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.basic_mask = basic_mask
        self.labels = labels


class trex_rcInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, subj_special_start_id, obj_special_start_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.subj_special_start_id = subj_special_start_id
        self.obj_special_start_id = obj_special_start_id


class custom_rcInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class trex_etInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, subj_special_start_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.subj_special_start_id = subj_special_start_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding="utf8") as f:
            return json.load(f)


class FindHeadInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class find_head_InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        # self.indexes = indexes


class MLMProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type=None):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train", data_dir)

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, data_dir
        )

    def get_labels(self):
        """See base class."""
        return

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line["token"]
            label = line["label"]

            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


class FindHeadProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type=None):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train", data_dir)

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, data_dir
        )

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line["sent"]
            text_b = (line["tokens"], line["pairs"])

            examples.append(FindHeadInputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples


def convert_examples_to_features_find_head(
    examples,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        try:
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            # start, end = example.text_b[0], example.text_b[1]
            sentence = example.text_a
            indexes, pairs = example.text_b

            tail_index2father_index = {pair["dependent_index"]: pair["governor_index"] for i, pair in enumerate(pairs)}
            labels = []  # fathers,heads
            for i in range(len(indexes)):
                labels.append(tail_index2father_index[(i + 1)])
            word_labels = copy.deepcopy(labels)

            tokens = [cls_token]
            sub_word_indexes = [1]  # count from 0，[cls] index=1
            length = 1  # as for [CLS]

            for i, index in enumerate(indexes):
                start = index[str(i + 1)]["start"]
                end = index[str(i + 1)]["end"]
                token = sentence[start:end]
                pbe_token = tokenizer.tokenize(token)
                tokens += pbe_token
                sub_word_indexes.append(length + 1)
                length += len(pbe_token)

            tokens += [sep_token]
            word_index2subword_index = {(i): sub_word_indexes[i] for i in range(len(indexes))}

            for i, label in enumerate(labels):
                labels[i] = word_index2subword_index[label]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if len(tokens) >= max_seq_length:
                continue
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            labels = [0] + labels + [0]
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            padding_length_label = max_seq_length - len(labels)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                labels = ([0] * padding_length_label) + labels
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
                labels = labels + ([0] * padding_length_label)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if output_mode == "classification":
                label_id = labels
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("sentence: %s" % (sentence))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("word_labels: %s" % word_labels)

                logger.info("sub_word_indexes: %s" % sub_word_indexes)
                logger.info("sub_wordlabels: %s" % (labels))

            features.append(
                find_head_InputFeatures(
                    input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id
                )
            )
        except:
            continue
    return features


relations = [
    "per:siblings",
    "per:parents",
    "org:member_of",
    "per:origin",
    "per:alternate_names",
    "per:date_of_death",
    "per:title",
    "org:alternate_names",
    "per:countries_of_residence",
    "org:stateorprovince_of_headquarters",
    "per:city_of_death",
    "per:schools_attended",
    "per:employee_of",
    "org:members",
    "org:dissolved",
    "per:date_of_birth",
    "org:number_of_employees/members",
    "org:founded",
    "org:founded_by",
    "org:political/religious_affiliation",
    "org:website",
    "org:top_members/employees",
    "per:children",
    "per:cities_of_residence",
    "per:cause_of_death",
    "org:shareholders",
    "per:age",
    "per:religion",
    "no_relation",
    "org:parents",
    "org:subsidiaries",
    "per:country_of_birth",
    "per:stateorprovince_of_death",
    "per:city_of_birth",
    "per:stateorprovinces_of_residence",
    "org:country_of_headquarters",
    "per:other_family",
    "per:stateorprovince_of_birth",
    "per:country_of_death",
    "per:charges",
    "org:city_of_headquarters",
    "per:spouse",
]


def convert_examples_to_features_entity_mlm(
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = example.text_a
        tags = example.label
        entities_pos = _get_entity_pos(tags)
        tokenization = tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
            max_length=max_seq_length,
            truncation="longest_first",
        )

        word_ids = tokenization.word_ids()

        word_ids = torch.tensor([word_id if word_id is not None else -1 for word_id in word_ids])
        input_ids = tokenization["input_ids"].squeeze()
        labels = input_ids.detach().clone()
        attention_mask = tokenization["attention_mask"].squeeze()

        # for this first version, we will only use masks, just like in https://arxiv.org/abs/1905.07129
        # we could also try to swap some entities with random ones, to make our model learn how to disentangle all that.
        basic_mask = torch.rand(input_ids.size()) < 0.15  # basic mask for MLM. Could play with this value

        # TODO: find a fully vectorized solution for this? Idk if this is possible.
        entity_masks = []
        for pos in entities_pos:
            tmp1 = word_ids >= pos[0]
            # print(tmp1)
            tmp2 = word_ids <= pos[1]
            # print(tmp2)
            ent_mask = torch.logical_and(tmp1, tmp2)  # boolean tensor for the entity
            # print(tmp)
            # entity_masks.append(ent_mask)
            if torch.logical_and(basic_mask, ent_mask).any():
                # if any token from the entity is chosen, mask the whole entity
                basic_mask[ent_mask] = True
            entity_masks.append(ent_mask)
        entity_masks = sum(entity_masks)

        input_ids[basic_mask] = tokenizer.mask_token_id  # masking all picked tokens

        assert len(input_ids.tolist()) == max_seq_length
        assert len(attention_mask.tolist()) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in basic_mask]))
        #     logger.info("labels: %s" % " ".join([str(x) for x in labels]))
        features.append(
            mlmInputFeatures(
                input_ids=input_ids.tolist(),
                input_mask=attention_mask.tolist(),
                basic_mask=basic_mask.tolist(),
                labels=labels.tolist(),
            )
        )
    return features


def _get_entity_pos(tags):
    entities_pos = []
    cont = 0
    i_max = len(tags)
    for i, tag in enumerate(tags):
        if tag != "O" and cont == 0:
            index = i  # remember the
            if i == (i_max - 1):
                # case where entity is one word long and at the end
                entities_pos.append((index, index + cont))
            cont += 1
        elif cont != 0 and tag != "O" and i != (i_max - 1):
            # continuing an entity case
            cont += 1
        elif cont != 0 and tag == "O":
            # standard case
            entities_pos.append((index, index + cont - 1))
            cont = 0
        elif cont != 0 and i == (i_max - 1):
            # case where entity is more than one word long and at the end
            entities_pos.append((index, index + cont))
    return entities_pos


def convert_examples_to_features_entity_typing(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        start, end = example.text_b[0], example.text_b[1]
        sentence = example.text_a
        tokens_0_start = tokenizer.tokenize(sentence[:start])
        tokens_start_end = tokenizer.tokenize(sentence[start:end])
        tokens_end_last = tokenizer.tokenize(sentence[end:])
        tokens = (
            [cls_token]
            + tokens_0_start
            + tokenizer.tokenize("@")
            + tokens_start_end
            + tokenizer.tokenize("@")
            + tokens_end_last
            + [sep_token]
        )
        start = 1 + len(tokens_0_start)
        end = 1 + len(tokens_0_start) + 1 + len(tokens_start_end)

        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
        start_id = np.zeros(max_seq_length)
        start_id[start] = 1
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                start_id=start_id,
            )
        )
    return features


# trex_relations = ['P178', 'P17', 'P569', 'P27', 'P36', 'P150', 'P1376', 'P47', 'P501', 'P131', 'P159', 'P103', 'P127',
#                   'P138', 'P118', 'P831', 'P664', 'P54', 'P115', 'P279', 'P681', 'P527', 'P106', 'P570', 'P57', 'P58',
#                   'P162', 'P344', 'P86', 'P1040', 'P161', 'P40', 'P22', 'P3373', 'P26', 'P25', 'P37', 'P1412', 'P364',
#                   'P495', 'P449', 'P105', 'P20', 'P451', 'P735', 'P101', 'P136', 'P1441', 'P112', 'P706', 'P175',
#                   'P641', 'P463', 'P1001', 'P155', 'P800', 'P171', 'P361', 'P39', 'P530', 'P19', 'P69', 'P194', 'P137',
#                   'P607', 'P1313', 'P1906', 'P205', 'P264', 'P466', 'P170', 'P1431', 'P580', 'P582', 'P740', 'P50',
#                   'P2578', 'P2579', 'P156', 'P30', 'P176', 'P166', 'P241', 'P413', 'P135', 'P195', 'P276', 'P180',
#                   'P921', 'P1408', 'P1066', 'P802', 'P1038', 'P53', 'P102', 'P937', 'P2341', 'P828', 'P1542', 'P488',
#                   'P807', 'P190', 'P611', 'P710', 'P2094', 'P765', 'P1454', 'P35', 'P577', 'P108', 'P172', 'P1303',
#                   'P282', 'P885', 'P403', 'P1346', 'P6', 'P1344', 'P366', 'P1416', 'P749', 'P737', 'P559', 'P598',
#                   'P609', 'P1923', 'P1071', 'P179', 'P870', 'P140', 'P647', 'P1336', 'P1435', 'P1876', 'P619', 'P123',
#                   'P87', 'P747', 'P425', 'P551', 'P1532', 'P122', 'P461', 'P206', 'P585', 'P571', 'P931', 'P452',
#                   'P355', 'P306', 'P915', 'P1589', 'P460', 'P1387', 'P81', 'P400', 'P1142', 'P1027', 'P2541', 'P682',
#                   'P209', 'P674', 'P676', 'P199', 'P1889', 'P275', 'P974', 'P407', 'P1366', 'P412', 'P658', 'P371',
#                   'P92', 'P1049', 'P750', 'P38', 'P277', 'P410', 'P1268', 'P469', 'P200', 'P201', 'P1411', 'P272',
#                   'P157', 'P2439', 'P61', 'P1995', 'P509', 'P287', 'P16', 'P121', 'P1002', 'P2388', 'P2389', 'P2632',
#                   'P1343', 'P2936', 'P2789', 'P427', 'P119', 'P576', 'P2176', 'P1196', 'P263', 'P1365', 'P457', 'P1830',
#                   'P186', 'P832', 'P141', 'P2175', 'P840', 'P1877', 'P1056', 'P3450', 'P1269', 'P113', 'P533', 'P3448',
#                   'P1191', 'P927', 'P610', 'P1327', 'P177', 'P1891', 'P169', 'P2670', 'P793', 'P770', 'P3137', 'P1383',
#                   'P1064', 'P134', 'P945', 'P84', 'P1654', 'P2522', 'P1552', 'P1037', 'P286', 'P144', 'P689', 'P541',
#                   'P991', 'P726', 'P780', 'P397', 'P398', 'P149', 'P1478', 'P98', 'P500', 'P1875', 'P2554', 'P59',
#                   'P3461', 'P414', 'P748', 'P291', 'P85', 'P2348', 'P3320', 'P462', 'P1462', 'P2597', 'P2512', 'P1018',
#                   'P21', 'P208', 'P2079', 'P1557', 'P1434', 'P1080', 'P1445', 'P1050', 'P3701', 'P767', 'P1299', 'P126',
#                   'P360', 'P1304', 'P1029', 'P1672', 'P1582', 'P184', 'P2416', 'P65', 'P575', 'P3342', 'P3018', 'P183',
#                   'P2546', 'P2499', 'P2500', 'P408', 'P450', 'P97', 'P417', 'P512', 'P1399', 'P404', 'P822', 'P941',
#                   'P189', 'P725', 'P1619', 'P129', 'P629', 'P88', 'P2545', 'P1068', 'P1308', 'P1192', 'P2505', 'P376',
#                   'P1535', 'P708', 'P1479', 'P2283', 'P1962', 'P2184', 'P163', 'P1419', 'P2286', 'P3190', 'P790',
#                   'P1547', 'P1444', 'P504', 'P2596', 'P3095', 'P3300', 'P881', 'P1880', 'P358', 'P1427', 'P2438',
#                   'P523', 'P524', 'P826', 'P485', 'P3679', 'P437', 'P553', 'P66', 'P2650', 'P816', 'P517', 'P1072',
#                   'P78', 'P415', 'P825', 'P1302', 'P1716', 'P411', 'P734', 'P110', 'P1264', 'P289', 'P421', 'P2238',
#                   'P375', 'P2989', 'P669', 'P2289', 'P111', 'P197', 'P620', 'P467', 'P3712', 'P185', 'P841', 'P739',
#                   'P3301', 'P568', 'P567', 'P479', 'P625', 'P1433', 'P1429', 'P880', 'P1414', 'P547', 'P1731', 'P618',
#                   'P2978', 'P1885', 'P516', 'P556', 'P522', 'P237', 'P1809', 'P2098', 'P1322', 'P3764', 'P2633',
#                   'P1312', 'P859', 'P114', 'P2962', 'P1073', 'P1000', 'P1158', 'P196', 'P520', 'P2155', 'P606', 'P3403',
#                   'P720']

trex_relations = [
    "no_relation",
    "facet",
    "counseling",
    "anesthesia",
    "treatment",
    "diagnostics",
    "profession",
    "consultation",
    "surgical procedures",
    "childbirth",
    "education",
    "technical act",
    "medical documents",
    "prescription",
    "product",
]


class TREXProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample
        )

    def get_dev_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample
        )

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return trex_relations

    def _create_examples(self, lines, dataset_type, negative_sample):
        """Creates examples for the training and dev sets."""
        examples = []
        no_relation_number = negative_sample
        for (i, line) in enumerate(lines):
            guid = i
            # text_a: tokenized words
            text_a = line["token"]
            # text_b: other information
            text_b = (line["subj_start"], line["subj_end"], line["obj_start"], line["obj_end"])
            label = line["relation"]
            if label == "no_relation" and dataset_type == "train":
                no_relation_number -= 1
                if no_relation_number > 0:
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                    continue
            else:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class CustomProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample
        )

    def get_dev_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample
        )

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        custom_relations = [
            "range",
            "alt label",
            "can be service given by institution organization",
            "can be person doing intervention",
            "is condition treated by institution organization",
            "inverse of",
            "treats",
            "is drug treats condition",
            "is discipline treats sign symptom",
            "is technology used to perform treatment",
            "abbreviation",
            "is anatomy affected by condition",
            "is partof",
            "is condition treated by discipline",
            "has primary target population",
            "hidden label",
            "domain",
            "is institution organization treating condition",
            "is condition treated by intervention",
            "is intervention done by institution organization",
            "scope",
            "is discipline treats condition",
            "disjoint with",
            "is sign symptom of condition",
            "can be condition treated by institution organization",
            "can have target population",
            "comment",
            "is subspecialty of discipline",
            "is discipline of subspecialty",
            "is service given by institution organization",
            "can be institution organization treating condition",
            "is intervention done by person",
            "is drug prescribed by person",
            "can be intervention done by person",
            "feminin",
            "studies",
            "is person of discipline",
            "is condition treated by drug",
            "pref label",
            "is person does intervention",
            "is discipline of person",
        ]
        return custom_relations

    def _create_examples(self, lines, dataset_type, negative_sample):
        """Creates examples for the training and dev sets."""
        examples = []
        no_relation_number = negative_sample
        guid = 0
        for subject, relations in lines.items():
            # text_a: tokenized words
            text_a = subject
            # text_b: other information
            for predicate, object in relations["relations"]:
                guid += 1
                text_b = object
                label = predicate
                if label == "disjoint with" and dataset_type == "train":
                    no_relation_number -= 1
                    if no_relation_number > 0:
                        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    else:
                        continue
                else:
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a = example.text_a
        text_b = example.text_b

        if isinstance(tokenizer, RobertaTokenizer):
            tokenized = tokenizer(text_a + "</s></s>" + text_b, max_length=max_seq_length, padding="max_length")
        else:
            tokenized = tokenizer.encode(text=text_a, text_pair=text_b, max_length=max_seq_length, padding="max_length")

        input_ids = tokenized["input_ids"]
        input_mask = tokenized["attention_mask"]

        if "segment_ids" in tokenized.keys():
            segment_ids = tokenized["segment_ids"]
        else:
            segment_ids = [sequence_a_segment_id] * len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(label_map[example.label])
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))

        features.append(
            custom_rcInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def convert_examples_to_features_trex(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a = example.text_a
        subj_start, subj_end, obj_start, obj_end = example.text_b
        # relation = example.label
        if subj_start < obj_start:
            tokens = tokenizer.tokenize(" ".join(text_a[:subj_start]))
            subj_special_start = len(tokens)
            tokens += ["@"]
            tokens += tokenizer.tokenize(" ".join(text_a[subj_start : subj_end + 1]))
            tokens += ["@"]
            tokens += tokenizer.tokenize(" ".join(text_a[subj_end + 1 : obj_start]))
            obj_special_start = len(tokens)
            tokens += ["#"]
            tokens += tokenizer.tokenize(" ".join(text_a[obj_start : obj_end + 1]))
            tokens += ["#"]
            tokens += tokenizer.tokenize(" ".join(text_a[obj_end + 1 :]))
        else:
            tokens = tokenizer.tokenize(" ".join(text_a[:obj_start]))
            obj_special_start = len(tokens)
            tokens += ["#"]
            tokens += tokenizer.tokenize(" ".join(text_a[obj_start : obj_end + 1]))
            tokens += ["#"]
            tokens += tokenizer.tokenize(" ".join(text_a[obj_end + 1 : subj_start]))
            subj_special_start = len(tokens)
            tokens += ["@"]
            tokens += tokenizer.tokenize(" ".join(text_a[subj_start : subj_end + 1]))
            tokens += ["@"]
            tokens += tokenizer.tokenize(" ".join(text_a[subj_end + 1 :]))

        _truncate_seq_pair(tokens, [], max_seq_length - 2)
        tokens = ["<s>"] + tokens + ["</s>"]
        subj_special_start += 1
        obj_special_start += 1

        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(label_map[example.label])
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))

        if subj_special_start >= max_seq_length:
            continue
            # subj_special_start = max_seq_length - 10
        if obj_special_start >= max_seq_length:
            continue
            # obj_special_start = max_seq_length - 10

        subj_special_start_id = np.zeros(max_seq_length)
        obj_special_start_id = np.zeros(max_seq_length)
        subj_special_start_id[subj_special_start] = 1
        obj_special_start_id[obj_special_start] = 1

        features.append(
            trex_rcInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                subj_special_start_id=subj_special_start_id,
                obj_special_start_id=obj_special_start_id,
            )
        )
    return features


class TREXProcessor_et(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample
        )

    def get_dev_examples(self, data_dir, dataset_type, negative_sample):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type, negative_sample
        )

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return trex_relations_et

    def _create_examples(self, lines, dataset_type, negative_sample):
        """Creates examples for the training and dev sets."""
        examples = []
        no_relation_number = negative_sample
        for (i, line) in enumerate(lines):
            guid = i
            # text_a: tokenized words
            text_a = line["token"]
            # text_b: other information
            # text_b = (line['word_start'], line['word_end'])
            text_b = (line["subj_start"], line["subj_end"])
            label = line["subj_label"]
            if label == "no_relation" and dataset_type == "train":
                no_relation_number -= 1
                if no_relation_number > 0:
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                    continue
            else:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "custom": CustomProcessor,
    "trex": TREXProcessor,
    "trex_entity_typing": TREXProcessor_et,
    "find_head": FindHeadProcessor,
    "mlm": MLMProcessor,
}

output_modes = {
    "custom": "classification",
    "trex": "classification",
    "trex_entity_typing": "classification",
    "find_head": "classification",
    "mlm": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "entity_type": 9,
}
