import numpy as np
from torch.utils.data import Dataset

# from copy import deepcopy
# from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
# from copy import deepcopy
# import csv
# import scipy
from utils.utils import *

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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer


import torch.nn.functional as F





def reduce_pos_tag(s):
    if 'NN' in s:
        return 'NN'
    elif 'VB' in s:
        return 'VB'
    elif 'JJ' in s:
        return 'JJ'
@dataclass
class CustomCollatorRET():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    has_concepts: bool = False

    def __call__(self, features):
        # features=deepcopy(feats)

        encoder_features = [{'input_ids': feat['input_ids'],
                             'attention_mask': feat['attention_mask'],
                             } for feat in features]
        input_features = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        input_features.update({
            'sample_ids': [feat['sample_id'] for feat in features]
        })

        return input_features


class MyDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        """========Init========="""
        self.instances = data

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance

def gpu_setup(use_gpu=True, gpu_id="0"):  # , use_random_available=True
    print("\nSetting up GPU")
    # if len(gpu_id):
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # print("visibes",os.environ["CUDA_VISIBLE_DEVICES"])
    num_gpus = 1
    if torch.cuda.is_available() and use_gpu:
        print(f"{torch.cuda.device_count()} GPU available")
        # print('cuda available with GPU:', torch.cuda.get_device_name(0))

        # use all
        device = torch.device("cuda")

    else:
        if use_gpu and not torch.cuda.is_available():
            print('cuda not available')
        device = torch.device("cpu")

    print("Device is set to", device)
    return device

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_word_embedding_matrix(words, model, tokenizer, prep_batch_size=40, args=None, cache_path=None, is_sbert=False):
    """
    Get the word embedding matrix for a list of words.
    """
    # Get the embedding matrix for the words in the list.
    device = gpu_setup(use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    if path_exists(cache_path) and args.save_mat:
        print("loaded from ",cache_path)
        res_mat=torch.tensor(np.load(cache_path), dtype=torch.float)
        res_mat=res_mat.to("cpu")
        return res_mat

    max_seq_len = tokenizer.model_max_length if tokenizer.model_max_length <= 100000 else 512  # TODO
    print("max_seq_len",max_seq_len)
    batch_tokens = tokenizer(words, max_length=max_seq_len, truncation=True,padding=True)  # , return_tensors="pt" padding=True,
    inputs = [{"sample_id": j,
               "input_ids": batch_tokens["input_ids"][j],
               "attention_mask": batch_tokens["attention_mask"][j]}
              for j in range(len(batch_tokens["input_ids"]))]

    data_collator = CustomCollatorRET(tokenizer, model=model, max_length=max_seq_len, has_concepts=False)
    ds_dataloader = DataLoader(MyDataset(inputs), batch_size=args.prep_batch_size, shuffle=False, collate_fn=data_collator, drop_last=False)

    # embedding_matrix=model(**batch_tokens).last_hidden_state[:, 0, :]
    model=model.to(device)
    model.eval()



    logits_lists = []
    for batch in tqdm(ds_dataloader):
        for k, v in batch.items():
            if k != "sample_ids" and v is not None:
                batch[k] = v.to(device)#cuda()

        with torch.no_grad():
            logits = model(input_ids=batch[f"input_ids"],
                           attention_mask=batch[f"attention_mask"],
                           return_dict=True)
            # logits_lists.extend(logits.tolist())
        if not is_sbert:
            logits=logits.last_hidden_state[:,0,:]
            # Perform pooling
        if is_sbert:
            logits = mean_pooling(logits, batch['attention_mask'])

            # Normalize embeddings
            logits = F.normalize(logits, p=2, dim=1)

        logits_lists.append(logits.cpu())


    embedding_matrix = torch.cat(logits_lists, dim=0)

    np.save(cache_path, embedding_matrix.numpy())

    model=model.cpu()

    return embedding_matrix.to("cpu")


def get_consine_sim_matrix(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    # print(res)
    return res

# def get_consine_sim_matrix2(sents1, sents2):
#     a_norm = a / a.norm(dim=1)[:, None]
#     b_norm = b / b.norm(dim=1)[:, None]
#     res = torch.mm(a_norm, b_norm.transpose(0, 1))
#     # print(res)
#     return res


def get_candidate_pool(word, record):
    pool = []
    lemmatizer = WordNetLemmatizer()
    lemm=lemmatizer.lemmatize(word)
    if word in record:
        return record[word]
    for item in wn.synsets(word, pos=wn.NOUN)[:1]: #[:1]
        if lemm in item.name():
            for p in item.hypernyms():
                for c in p.hyponyms():
                    # print(c, p)
                    pool.extend([str(lemma.name()) for lemma in c.lemmas()])
    record[word] = pool
    return record[word]

def get_candidate_pool2(word, record, return_random=False, random_size=3):
    # assume word is a lemma
    pool = []
    # lemmatizer = WordNetLemmatizer()
    # lemm=lemmatizer.lemmatize(word)
    if word in record:
        return record[word] if not return_random else np.random.choice(record[word], size=min(len(record[word]), random_size), replace=False)
    for item in wn.synsets(word, pos=wn.NOUN)[:1]: #[:1]
        if word in item.name():
            for c in item.hyponyms():
                # print(c, p)
                pool.extend([str(lemma.name()).replace("_"," ") for lemma in c.lemmas()])
            for c in item.hypernyms():
                # print(c, p)
                pool.extend([str(lemma.name()).replace("_"," ") for lemma in c.lemmas()])

    record[word] = pool
    return record[word] if not return_random else np.random.choice(record[word], size=min(len(record[word]), random_size),  replace=False)

def compute_token2nodepos_batch(cur_ranges, bsz, seqlen, accumulate=False):
    token2nodeid = -torch.ones(bsz, seqlen, dtype=torch.long)
    start_pos=0
    for batch_id, batch_range in enumerate(cur_ranges):
        for node_id, (s, e) in enumerate(batch_range):
            token2nodeid[batch_id, s:e] = node_id if not accumulate else start_pos + node_id
        start_pos += len(batch_range)
    return token2nodeid.long()


def compute_token2nodepos(cur_ranges, seqlen, pad_mask=None):
    token2nodeid = -torch.ones(seqlen, dtype=torch.long)
    for node_id, (s, e) in enumerate(cur_ranges):
        if pad_mask is not None and pad_mask[node_id] == 0:  # padding for entities
            continue
        token2nodeid[s:e] = node_id
    return token2nodeid.long()

def flatten_list(ls_batch):
    return [item for sublist in ls_batch for item in sublist]

def token_lens_to_idxs(batch_list_of_ranges):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """

    # input is b x node (uneven)
    res = []

    max_token_num = max([len(b) for b in batch_list_of_ranges])
    max_token_len = max([(e - s) for b in batch_list_of_ranges for s, e in b])
    idxs, masks = [], []
    for b in batch_list_of_ranges:
        seq_idxs, seq_masks = [], []
        offset = 0
        for s, e in b:
            token_len = e - s
            seq_idxs.extend([i for i in range(s, e)]
                            + [0] * (max_token_len - token_len))  # -1
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
        seq_idxs.extend([0] * max_token_len * (max_token_num - len(b)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(b)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    # max_token_num = max([len(x) for x in token_lens])
    # max_token_len = max([max(x) for x in token_lens])
    # idxs, masks = [], []
    # for seq_token_lens in token_lens:
    #     seq_idxs, seq_masks = [], []
    #     offset = 0
    #     for token_len in seq_token_lens:
    #         seq_idxs.extend([i + offset for i in range(token_len)]
    #                         + [-1] * (max_token_len - token_len))
    #         seq_masks.extend([1.0 / token_len] * token_len
    #                          + [0.0] * (max_token_len - token_len))
    #         offset += token_len
    #     seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
    #     seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
    #     idxs.append(seq_idxs)
    #     masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def merge_all_sents(sents, tokenizer=None, step_types_dict=None, step_types=None, has_cls=True):
    merged_sent = [tokenizer.cls_token] if has_cls else []
    start_pos = 1 if has_cls else 0   # cls existss
    start_positions = [start_pos]
    for i, sent in enumerate(sents):
        merged_sent += (sent + [step_types_dict[step_types[i]]])
        start_pos = len(merged_sent)
        start_positions.append(start_pos)
    return merged_sent, start_positions


def accumulate_ranges(ranges_batch, start_positions):
    for m, ranges in enumerate(ranges_batch):
        for i, (k, s, e) in enumerate(ranges):
            # res.append((s+start_positions[i], start_positions[i + 1]))
            ranges[i] = [s + start_positions[k], e + start_positions[k]]
    return ranges_batch


def update_ranges_with_subtoken_mapping(ranges_batch, new_map, max_seq_length, filter_oor=False):
    valid_range_mask = []
    res = []
    for i, ranges in enumerate(ranges_batch):
        tmp = []
        tmp2 = []
        for j, (s, e) in enumerate(ranges):
            if e not in new_map:
                breakpoint()
                embed()
            if new_map[e] < max_seq_length - 1:  # node_cnt == max_node_num
                tmp.append([new_map[s], new_map[e]])
                tmp2.append(1)
            else:
                if not filter_oor:
                    tmp.append([-1, -1])
                tmp2.append(0)
        if not filter_oor or len(tmp) > 1:
            res.append(tmp)
        valid_range_mask.append(tmp2)
    return res, valid_range_mask


def exclude_columns_for_lists(ls_batch, cols):
    for i, ls in enumerate(ls_batch):
        for j, l in enumerate(ls):
            ls[j] = [x for k, x in enumerate(l) if k not in cols]
    return ls_batch

def aggregate_graphs(graphs, node_span_batch, valid_node_masks, max_node_num):
    """

    :param graphs: list of graphs, [[[0,1], [1,0]]
    :example:
    :param valid_node_masks: [[-1,1,1,-1]]
    :param max_node_num:
    :return:
    """
    """====Convert node ranges to new token indices===="""
    graph_instances = []
    node_cnt = 0

    for i, (n_mask, graph) in enumerate(zip(valid_node_masks, graphs)):

        node_spans = node_span_batch[i]
        edge_index = np.array(graph['edge_index'])
        edge_attr = graph['edge_attr']

        num_valid_nodes=sum(n_mask)
        # num_invalid_nodes=len(n_mask) - num_valid_nodes
        # e_mask = [0 if (n_mask[s] == 0 or n_mask[e] == 0) else 1 for s, e in edge_index]
        # num_valid_edges = sum(e_mask)
        # num_invalid_edges = len(e_mask) - num_valid_edges
        if num_valid_nodes + node_cnt > max_node_num:
            break

        if num_valid_nodes == len(n_mask):
            # graph_instances.append(graph)
            # root_indices.append(node_cnt)
            node_cnt += len(n_mask) # num_valid_nodes
            graph_instances.append(Data(x=get_tensor_long(node_spans),
                                        edge_index=get_tensor_long(edge_index.T),
                                        edge_attr=get_tensor_long(edge_attr)))

    if not graph_instances:
        graph_instances.append(Data(x=get_tensor_long([]), edge_index=get_tensor_long([]), edge_attr=get_tensor_long([])))
        print(f"empty graph")
        embed()
    return Batch.from_data_list(graph_instances)  # , root_indices


def sents_to_token_ids_accumulate(sents=None, max_seq_length=None, tokenizer=None,
                                  special_tks=None, step_types_dict=None, use_special_tag=False, list_of_ranges=None, list_of_ent_ranges=None, step_types=None, max_node_num=200):
    # print("\n\nsents_to_token_ids_with_graph")
    """====update word indices after merging all steps===="""
    # special_tks = ["[GOAL]", "[SUBGOAL]", "[STEP]", tokenizer.sep_token, "<ROOT>"]
    #
    # # g is like x, edge index, edge attr
    # step_types_dict = {
    #     "goal": "[GOAL]",
    #     "subgoal": "[SUBGOAL]",
    #     "event": "[STEP]",
    # }
    #
    # if not use_special_tag:
    #     step_types_dict = {k: tokenizer.sep_token for k, v in step_types_dict.items()}

    merged_sent = [tokenizer.cls_token]
    start_pos = 1  # cls existss
    for i, ent2ranges in enumerate(list_of_ent_ranges):
        # ranges=
        for ent, ranges in ent2ranges.items():
            if len(ranges):  # empty graph
                # assert len(ent2ranges[ent])==2
                # if -1 in ent2ranges[ent][1]: ent2ranges[ent].pop()
                ent2ranges[ent] = (np.array(ranges) + start_pos).tolist()

        # if len(ranges): # empty graph
        #     list_of_ranges[i]=(np.array(ranges) + start_pos).tolist()
        merged_sent += (sents[i] + [step_types_dict[step_types[i]]])
        start_pos = len(merged_sent)

    input_ids, new_map, _, _, attention_mask = sent_to_token_ids(merged_sent, max_seq_length, tokenizer, shift_right=False, add_sep_at_end=True, has_cls_at_start=True,
                                                                 special_tks=special_tks)

    out_of_range = False
    node_cnt = 0

    for i, ent2ranges in enumerate(list_of_ent_ranges):
        # ranges=
        # already out of range
        if out_of_range:
            list_of_ent_ranges[i] = {}
            continue
        for ent, ranges in list(ent2ranges.items()):
            for j, (s, e) in enumerate(ranges):
                if e not in new_map:
                    embed()
                if new_map[e] >= max_seq_length - 1 or node_cnt == max_node_num:
                    out_of_range = True
                    ent2ranges.pop(ent)
                    break
                ent2ranges[ent][j][0] = new_map[s]
                ent2ranges[ent][j][1] = new_map[e]
                node_cnt += 1
    return input_ids, attention_mask, list_of_ent_ranges  # list_of_ranges


def sent_to_token_ids(sent, max_seq_length, tokenizer, shift_right=False, add_sep_at_end=True, has_cls_at_start=True, special_tks=None, end_token=None):
    """
    @param sent: list of tokens, with cls
    @param ent_pos_list: list of s e index pairs for each mention, like [[0,1],[5,7]]
    @param max_seq_length: max bert seqlen
    @param tokenizer: tokenizer
    @param shift_right: always set true, shift new mention position in tokens +1 because we have CLS
    @param add_marker: add * to before and after mention
    @return: list of tokens, and updated ent_pos_list
assume no sep
    """

    new_map = {}
    original_pos_vec = []
    sents = []

    if shift_right:
        front_token = tokenizer.cls_token
        if front_token is None: front_token = tokenizer.bos_token
        if front_token is None: front_token = tokenizer.pad_token  # T5
        sents = [front_token] + sent

    for i_t, token in enumerate(sent):
        token = token.strip()
        if not len(token):
            token = " "  # prevent empty token to disappear, making

        # tokens_wordpiece = tokenizer.tokenize(token)
        after_the_first_token = (has_cls_at_start and i_t > 1) or (not has_cls_at_start and i_t > 0)
        if token in special_tks or (after_the_first_token and sent[i_t - 1].strip() in special_tks):  # for gpt like tokenizer which cares about space
            tokens_wordpiece = tokenizer.tokenize(token)
        else:
            tokens_wordpiece = tokenizer.tokenize((" " + token) if after_the_first_token else token)

        # if tokenizer.sep_token == token or (i_t > 1 and sent[
        #     i_t - 1].strip() == tokenizer.sep_token):  # for gpt like tokenizer which cares about space
        #     tokens_wordpiece = tokenizer.tokenize(token)
        # else:
        #     tokens_wordpiece = tokenizer.tokenize(
        #         " " + token if ((has_cls_at_start and i_t > 1) or (not has_cls_at_start and i_t > 0)) else token)
        new_map[i_t] = len(sents)
        for _ in range(len(tokens_wordpiece)):
            original_pos_vec.append(i_t)
        sents.extend(tokens_wordpiece)
    new_map[i_t + 1] = len(sents)

    # sents = sents[:max_seq_length - 2]
    if has_cls_at_start:
        sents = sents[:max_seq_length - 1]
    else:
        sents = sents[:max_seq_length - 2]
    if add_sep_at_end:
        # end_token = tokenizer.sep_token
        if end_token is None: end_token = tokenizer.eos_token

        sents += [end_token]
        # new_map[i_t + 2] = len(sents) # no need since sep will be moved left is sent too long

        original_pos_vec.append(original_pos_vec[-1] + 1)

    input_ids = tokenizer.convert_tokens_to_ids(sents)
    # print("sents",sents)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    return input_ids, new_map, original_pos_vec, token_type_ids, attention_mask, sents


def modify_output(s, tokenizer, is_tgt=False):
    tmp = s.strip()
    if tokenizer.cls_token is not None:
        tmp=tmp.replace(tokenizer.cls_token, "")
    tmp=tmp.replace(tokenizer.pad_token, "")

    # if tokenizer.sep_token is not None:
    #     s, _ = tmp.split(tokenizer.sep_token)[0], "\n".join(tmp.split(tokenizer.sep_token)[1:])
    # else:
    #     s, _ = tmp.split(tokenizer.eos_token)[0], "\n".join(tmp.split(tokenizer.eos_token)[1:])



    if is_tgt:
        s = tmp.replace(". ", f"\n" + " " * 18)

    else:
        s = tmp.replace(tokenizer.sep_token, f" {tokenizer.sep_token}\n" + " " * 18)
        if '[GOAL]' in s:
            s=s.replace("[GOAL]", " [GOAL]\n" + " " * 18). \
                replace("[SUBGOAL]", " [SUBGOAL]\n" + " " * 18). \
                replace("[STEP]", " [STEP]\n" + " " * 18)
    return s.strip()




def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def clear_subgoals3(step_ids, step_db):
    steps = [step_db[id] for id in step_ids]
    step_types = [step["step_type"] for step in steps]
    step_ids_new = []
    for i, step_id in enumerate(step_ids):
        if step_types[i] == "subgoal":
            continue
        step_ids_new.append(step_id)
    return step_ids_new
def clear_subgoals2(steps):
    steps_new = []
    step_types_new = []
    for i, step in enumerate(steps):
        if step["step_type"] == "subgoal":
            continue
        steps_new.append(step)
    return steps_new
def clear_subgoals(steps, step_types):
    steps_new = []
    step_types_new = []
    for i, step in enumerate(steps):
        if step_types[i] == "subgoal":
            continue
        steps_new.append(step)
        step_types_new.append(step_types[i])
    return steps_new, step_types_new