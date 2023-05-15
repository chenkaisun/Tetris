import numpy as np
from torch.utils.data import Dataset

# from copy import deepcopy
# from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
# from copy import deepcopy
# import csv
# import scipy
from utils.utils import *
from utils.data_utils import *

if module_exists("torch_geometric"):
    from torch_geometric.data import Batch, Data
from copy import deepcopy, copy
from multiprocessing import Pool
from tqdm import tqdm
from train_utils import get_tensor_long, get_tensor_float
from collections import defaultdict
from dataclasses import dataclass

from typing import Any, List, Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers import PreTrainedTokenizerBase


def pad_auxiliary():
    pass
    #


def pad_labels(features, tokenizer, model_name, label_pad_token_id):  # ,padding_side,label_pad_token_id
    labels = [feature["labels"].copy() for feature in features] if "labels" in features[0].keys() else None
    # if "t5" in self.model_name:
    #     print("labels", labels)
    #     print("labels", self.tokenizer.batch_decode(labels))
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.

    if labels is not None:
        max_label_length = max(len(l) for l in labels)  # max len of decoder input ids

        ##* shift labels right by 1
        # if
        # if "t5" not in self.model_name:
        # for feature in features:
        #     feature["labels"] = feature["labels"][1:]
        if "t5" not in model_name and "gpt2" not in model_name:
            for j, lb in enumerate(labels):
                labels[j] = lb[1:]
        # print("self.label_pad_token_id",self.label_pad_token_id)
        padding_side = tokenizer.padding_side
        for j, lb in enumerate(labels):
            remainder = [label_pad_token_id] * (max_label_length - len(lb))
            labels[j] = (
                lb + remainder if padding_side == "right" else remainder + lb
            )
    return labels

    #


@dataclass
class CustomCollator():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    has_concepts: bool = False
    components: Optional[List] = None
    model_name: Optional[str] = None
    verbose: Optional[bool] = False
    use_special_tag: Optional[bool] = False
    do_cl: Optional[bool] = False
    args: Optional[Any] = None

    def get_padded_features(self, features, labels, is_decoder=False):

        """CLF or not decode"""
        prefix = 'decoder_' if is_decoder else ''
        decoder_features = [{'input_ids': feat[f'{prefix}input_ids'],
                             'attention_mask': feat[f'{prefix}attention_mask'],
                             # 'labels': labels[j],
                             } for j, feat in enumerate(features)]
        if is_decoder:
            if "labels" in features[0]:
                for j, feat in enumerate(features):
                    decoder_features[j]['labels'] = labels[j]
        decoder_features = self.tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        new_dict = {
            'decoder_input_ids': decoder_features.data['input_ids'],
            'decoder_attention_mask': decoder_features.data['attention_mask'],
            'labels': decoder_features.data['labels'],
        }
        decoder_features.data = new_dict
        return decoder_features

    def __call__(self, features):
        sample_ids = [feature["sample_id"] for feature in features]
        if self.verbose:
            print("sample id", sample_ids)
        # print("sample id", sample_ids)
        """labels"""
        if "clf_label" not in features[0] and "labels" in features[0]:
            # make sure label is same as decoder input
            # print("clf_label not in")

            # for feature in features:
            #     feature["labels"] = feature['decoder_input_ids']
            # if "t5" in self.model_name:
            #     feature["labels"] = [self.tokenizer.eos_token_id]+feature['decoder_input_ids']
            # else:
            #     feature["labels"] = feature['decoder_input_ids']

            labels = [feature["labels"].copy() for feature in features] if "labels" in features[0].keys() else None
            # if "t5" in self.model_name:
            #     print("labels", labels)
            #     print("labels", self.tokenizer.batch_decode(labels))
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.

            if labels is not None:
                max_label_length = max(len(l) for l in labels)  # max len of decoder input ids

                ##* shift labels right by 1
                # if
                # if "t5" not in self.model_name:
                # for feature in features:
                #     feature["labels"] = feature["labels"][1:]
                if "t5" not in self.model_name and "gpt2" not in self.model_name:
                    for j, lb in enumerate(labels):
                        labels[j] = lb[1:]
                padding_side = self.tokenizer.padding_side
                # print("self.label_pad_token_id",self.label_pad_token_id)
                for j, lb in enumerate(labels):
                    remainder = [self.label_pad_token_id] * (max_label_length - len(lb))
                    labels[j] = (
                        lb + remainder if padding_side == "right" else remainder + lb
                    )

        """encoder features"""
        encoder_features = [{'input_ids': feat['input_ids'],
                             'attention_mask': feat['attention_mask'],
                             } for feat in features]

        input_features = {}
        """agg features"""
        input_features.update(self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ))

        """CLF or not decode"""
        decoder_features = [{'input_ids': feat['decoder_input_ids'],
                             'attention_mask': feat['decoder_attention_mask'],
                             # 'labels': labels[j],
                             } for j, feat in enumerate(features)]
        if "labels" in features[0]:
            for j, feat in enumerate(features):
                decoder_features[j]['labels'] = labels[j]
        decoder_features = self.tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # print("input_features[labels]", input_features['labels'])
        if "gpt2" not in self.model_name:
            input_features['decoder_input_ids'] = decoder_features['input_ids']
            input_features['decoder_attention_mask'] = decoder_features['attention_mask']
        # else:
        #     if "gpt2" not in self.model_name:
        if "labels" in features[0]:
            input_features['labels'] = decoder_features['labels']

        """neg samples"""
        # if self.model.training and not  features[0]["in_train"] or not self.model.training and features[0]["in_train"]:
        #     print("training mode problem in collate")
        #     breakpoint()
        if "in_train" in features[0] and features[0]["in_train"] and self.do_cl:
            cl_sample_key = f"negative_samples"
            input_features["neg_samples"] = []
            for cl_sample_idx in range(len(features[0][cl_sample_key])):
                # neg_sample_features = [{'input_ids': feat[neg_sample_key][neg_sample_idx]['input_ids'],
                #                         'attention_mask': feat[neg_sample_key][neg_sample_idx]['attention_mask'],
                #                         } for feat in features]
                cl_sample_features = [feat[cl_sample_key][cl_sample_idx] for feat in features]
                cl_labels = pad_labels(cl_sample_features, self.tokenizer, self.model_name, self.label_pad_token_id)

                # for j, lb in enumerate(cl_labels):
                #     if len(lb) >= 2 and lb[0] == self.tokenizer.eos_token_id and lb[1] == self.label_pad_token_id:
                #         lb[0] = self.label_pad_token_id

                cl_decoder_features = self.get_padded_features(cl_sample_features, cl_labels, is_decoder=True)
                cl_decoder_features.data["cl_mask"] = torch.tensor([[(not feat["is_empty"])] for feat in cl_sample_features], dtype=torch.double)
                input_features["neg_samples"].append(cl_decoder_features)
                # if "labels" in features[0]:
                #     input_features['labels'] = cl_decoder_features['labels']
            input_features["is_eval"] = not features[0]["in_train"]

        # not used right now

        bsz, slen = input_features['input_ids'].shape

        if self.verbose:
            if -1 in sample_ids:
                breakpoint()
        """=============batch entities============="""
        if self.model_name == "see":
            entity_batch = deepcopy([feat['entities'] for feat in features])  # .copy()
            token_tags_batch = deepcopy([feat['token_tags'] for feat in features])  # .copy()
            token_tags_batch = [t + [0] * (slen - len(t)) for t in token_tags_batch]
            # max_num_ents=max([len(ent_list) for ent_list in ent_lists])
            # max_time_len=max([len(ent["mention_msk"]) for ent_list in ent_lists for ent in ent_list])
            """in entity ranges for lstm, only take 1 token for an entity, since multiple tokens will repeat from graph output"""
            """entity level padding"""
            """idxs in sample contains idx for for each entity"""

            for entities in entity_batch:
                entities['token2nodepos'] = compute_token2nodepos(cur_ranges=flatten_list(entities["ranges"]), seqlen=slen, pad_mask=entities['pad_mask'])
            input_features['entities'] = entity_batch

            """=============batch graphs============="""
            graph_list = [feat['text_graph'] for feat in features]
            ranges_batch = [g['graph'].x.tolist() for g in graph_list]
            idxs, masks, max_token_num, max_token_len = token_lens_to_idxs(ranges_batch)
            text_graphs = {}
            text_graphs.update({
                "idxs": get_tensor_long(idxs),
                "masks": get_tensor_float(masks),
                "max_token_num": max_token_num,
                "max_token_len": max_token_len,
            })

            text_graphs['token2nodepos'] = compute_token2nodepos_batch(cur_ranges=ranges_batch, bsz=bsz, seqlen=slen, accumulate=True)
            text_graphs['num_nodes'] = [len(x) for x in ranges_batch]  # num_nodes

            graph_instances = []
            for graph_data in graph_list:
                graph = graph_data["graph"]
                graph_instances.append(Data(x=graph.x,
                                            edge_index=graph.edge_index,
                                            edge_attr=graph.edge_attr))
            text_graphs['graph'] = Batch.from_data_list(graph_instances)
            text_graphs['token_tags_batch'] = get_tensor_long(token_tags_batch)

            input_features['text_graphs'] = text_graphs

        if "t5" in self.model_name:
            input_features.pop('decoder_input_ids')
            input_features.pop('decoder_attention_mask')
        # prepare decoder_input_ids
        if self.model is not None and "gpt2" not in self.model_name and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=input_features["labels"])
            input_features["decoder_input_ids"] = decoder_input_ids

        return input_features


@dataclass
class _CustomCollator():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    has_concepts: bool = False
    components: Optional[List] = None
    model_name: Optional[str] = None
    verbose: Optional[bool] = False
    use_special_tag: Optional[bool] = False

    def __call__(self, features):
        sample_ids = [feature["sample_id"] for feature in features]
        # print("sample id", sample_ids)

        """labels"""

        if "clf_label" not in features[0]:
            # make sure label is same as decoder input
            # print("clf_label not in")

            # for feature in features:
            #     feature["labels"] = feature['decoder_input_ids']
            # if "t5" in self.model_name:
            #     feature["labels"] = [self.tokenizer.eos_token_id]+feature['decoder_input_ids']
            # else:
            #     feature["labels"] = feature['decoder_input_ids']

            labels = [feature["labels"].copy() for feature in features] if "labels" in features[0].keys() else None
            # if "t5" in self.model_name:
            #     print("labels", labels)
            #     print("labels", self.tokenizer.batch_decode(labels))
            # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
            # same length to return tensors.

            if labels is not None:
                max_label_length = max(len(l) for l in labels)  # max len of decoder input ids

                ##* shift labels right by 1
                # if
                # if "t5" not in self.model_name:
                # for feature in features:
                #     feature["labels"] = feature["labels"][1:]
                for j, lb in enumerate(labels):
                    labels[j] = lb[1:]

                padding_side = self.tokenizer.padding_side
                # print("self.label_pad_token_id",self.label_pad_token_id)
                for j, lb in enumerate(labels):
                    remainder = [self.label_pad_token_id] * (max_label_length - len(lb))
                    labels[j] = (
                        lb + remainder if padding_side == "right" else remainder + lb
                    )
                # for j,feature in enumerate(features):
                #     remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                #     feature["labels"] = (
                #         feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                #     )

        """encoder features"""

        use_nbr_extra = "cbr" in self.components
        do_cat_nbs = "cat_nbs" in self.components
        do_match = "match" in self.components
        do_est = "est" in self.components

        # print("use_nbr_extra",use_nbr_extra)
        # print("do_cat_nbs",do_cat_nbs)

        encoder_features_aggregate = None
        if (use_nbr_extra and not do_cat_nbs) or do_match:
            # print("is cbr and not do_cat_nbs")

            encoder_features_aggregate = []
            encoder_features = [{'input_ids': feat['input_ids'],
                                 'attention_mask': feat['attention_mask'],
                                 } for feat in features]
            encoder_features_aggregate.extend(encoder_features)  ## collate both nbs text and original text
            for j in range(len(features[0]['nbs_input_ids'])):
                encoder_features_aggregate.extend([{'input_ids': feat['nbs_input_ids'][j],
                                                    'attention_mask': feat['nbs_attention_mask'][j],
                                                    } for feat in
                                                   features])  # selfselfselfself nbs1nbs1nbs1nbs1 nbs2nbs2nbs2nbs2 ...

            # print("encoder_features",encoder_features)

            # print("len(features[0]['nbs_input_ids'])",len(features[0]['nbs_input_ids']))
            # print("len(encoder_features_aggregate)", len(encoder_features_aggregate))
            # for feat in features:
            #     true_sample_indices.append(len(encoder_features))
            #     encoder_features.append({'input_ids': feat['input_ids'],
            #                             'attention_mask': feat['attention_mask'],
            #                              })
            #
            #     encoder_features_aggregate.extend([{'input_ids': feat['nbs_input_ids'][j],
            #                              'attention_mask': feat['nbs_attention_mask'][j],
            #                              } for j in range(len(feat['nbs_input_ids']))])
            # encoder_features_nbs.append(([{'input_ids': feat['nbs_input_ids'][j],
            #                          'attention_mask': feat['nbs_attention_mask'][j],
            #                          } for j in range(len(feat['nbs_input_ids']))]))

        elif use_nbr_extra and do_cat_nbs:
            # print("is cbr and  do_cat_nbs")

            encoder_features = [{'input_ids': feat['cat_input_ids'],
                                 'attention_mask': feat['cat_attention_mask'],
                                 } for feat in features]
        else:

            # print("self.tokenizer.decode(features[0]['input_ids'])", self.tokenizer.decode(features[0]['input_ids']))
            encoder_features = [{'input_ids': feat['input_ids'],
                                 'attention_mask': feat['attention_mask'],
                                 } for feat in features]

        input_features = {}
        """agg features"""
        if (use_nbr_extra and not do_cat_nbs) or do_match:
            # print('use_nbr_extra and not do_cat_nbs:')

            aggregated_features = self.tokenizer.pad(
                encoder_features_aggregate,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # input_features.update({
            #     'aggregate_ipids':aggregated_features['input_ids'],
            #     'aggregate_atmsk':aggregated_features['attention_mask'],
            # })
            # # print("input_features bf", input_features)
            # input_features['input_ids'] = input_features['aggregate_ipids'].index_select(0, torch.tensor(true_sample_indices, dtype=torch.long))
            # input_features['attention_mask'] = input_features['aggregate_atmsk'].index_select(0, torch.tensor(true_sample_indices, dtype=torch.long))
            # print("input_features af", input_features)

            # aggregated_features['input_ids']=aggregated_features['input_ids'].view(1+num_nbs_each,
            #                                                                        num_sents//(1+num_nbs_each),
            #                                                                        seq_len)
            #
            # print("len aggregated_features", len(aggregated_features))
            input_features['input_ids'] = aggregated_features['input_ids'][:len(features)]
            input_features['attention_mask'] = aggregated_features['attention_mask'][:len(features)]
            input_features['attention_mask'] = input_features['attention_mask'].float()

            input_features['nbs_ipids'] = aggregated_features['input_ids'][len(features):].long()
            input_features['nbs_atmsk'] = aggregated_features['attention_mask'][len(features):].float()

            total_num_nbs_batch, seq_len = input_features['nbs_ipids'].shape
            num_nbs_each_sample = len(features[0]['nbs_input_ids'])
            input_features['nbs_ipids'] = input_features['nbs_ipids'].view(num_nbs_each_sample,
                                                                           int(total_num_nbs_batch // num_nbs_each_sample), seq_len)
            input_features['nbs_atmsk'] = input_features['nbs_atmsk'].view(num_nbs_each_sample,
                                                                           int(total_num_nbs_batch // num_nbs_each_sample), seq_len)
            # #nbs x batch size x seqlen
            # print("")
            # print(self.tokenizer.batch_decode(input_features['input_ids']))
            # for jj in range(input_features['nbs_ipids'].shape[0]):
            #     print(self.tokenizer.batch_decode(input_features['nbs_ipids'][jj]))

            # print('nbs_msk bf', [feat['nbs_mask'] for feat in features])
            input_features['nbs_msk'] = get_tensor_float(np.array([feat['nbs_mask'] for feat in features]).T)  # nbs per sample x B
            # print('\nnbs_msk',input_features['nbs_msk'])
            #
            # breakpoint()

            # print('nbs_ipids', input_features['nbs_ipids'].shape)
            # print('nbs_atmsk', input_features['nbs_atmsk'].shape)

        else:
            input_features.update(self.tokenizer.pad(
                encoder_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            ))
            # input_features['input_ids']=[1,2,3]
            # input_features['input_ids']=[Data(x=[1,2,3])]
            # print("input_features in coll", input_features)
            # print("input_features for normal", input_features)

        """CLF or not decode"""
        if "clf_label" not in features[0]:
            decoder_features = [{'input_ids': feat['decoder_input_ids'],
                                 'attention_mask': feat['decoder_attention_mask'],
                                 'labels': labels[j],
                                 } for j, feat in enumerate(features)]
            decoder_features = self.tokenizer.pad(
                decoder_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # print("input_features[labels]", input_features['labels'])

            input_features['decoder_input_ids'] = decoder_features['input_ids']
            input_features['decoder_attention_mask'] = decoder_features['attention_mask']

            input_features['labels'] = decoder_features['labels']
            # print("labels", self.tokenizer.decode(input_features['labels'][0], skip_special_tokens=False))
        else:
            # print("clf in")

            tmp = self.tokenizer([feature["tgt_text"] for feature in features], truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            input_features['tgt_ipids'] = tmp['input_ids']
            input_features['tgt_atmsk'] = tmp['attention_mask']
            input_features['labels'] = torch.tensor([feature["clf_label"] for feature in features], dtype=torch.long)
        # not used right now

        # Data(x=)
        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=input_features["labels"])
            input_features["decoder_input_ids"] = decoder_input_ids

        return input_features
