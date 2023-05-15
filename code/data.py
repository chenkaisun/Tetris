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
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

from typing import Any, List, Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers import PreTrainedTokenizerBase

import nltk
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from constants import *

cur_concept2id = None
cur_id2concept = None


def pad_to_batch(batch_tokens, max_seq_len=99999, pad_token_id=0, max_batch_seq_len=None):
    if max_batch_seq_len is not None:
        max_len = max_batch_seq_len
        # batch_tokens=[item[:max_batch_seq_len] for item in batch_tokens] # fixed max batch seqlen
        b_tokens = [item[:max_batch_seq_len] for item in batch_tokens]
    else:
        max_len = max([len(f) for f in batch_tokens])
        b_tokens = batch_tokens
    # print("\nbatch_tokens", batch_tokens)
    if isinstance(pad_token_id, int):
        input_ids = [f + [pad_token_id for _ in range(max_len - len(f))] for f in b_tokens]  # * (max_len - len(f))
    elif isinstance(pad_token_id, list):
        # print("yes list")
        # print("pad_token_id", pad_token_id)
        input_ids = [f + [pad_token_id[i] for _ in range(max_len - len(f))] for i, f in
                     enumerate(b_tokens)]  # * (max_len - len(f))

    input_mask = [[1.0] * len(f) + [0.0] * (max_len - len(f)) for f in b_tokens]
    # print(np.array(input_ids, dtype=int).shape)

    # if max_len >= max_seq_len:
    #     # if isinstance(pad_token_id, int):
    #     input_ids, input_mask = np.array(input_ids, dtype=int)[:, :max_seq_len].tolist(), np.array(input_mask, dtype=int)[:,
    #                                                                                  :max_seq_len].tolist()

    # print("\ninput_ids", input_ids)
    return input_ids, input_mask
    return torch.tensor(input_ids, dtype=torch.long)[:, :max_seq_len], torch.tensor(input_mask, dtype=torch.float)[:,
                                                                       :max_seq_len]


def sent_with_entities_to_token_ids(sent, ent_pos_list, max_seq_length, tokenizer, shift_right=True, add_marker=True):
    """
    @param sent: list of tokens
    @param ent_pos_list: list of s e index pairs for each mention, like [[0,1],[5,7]]
    @param max_seq_length: max bert seqlen
    @param tokenizer: tokenizer
    @param shift_right: always set true, shift new mention position in tokens +1 because we have CLS
    @param add_marker: add * to before and after mention
    @return: list of tokens, and updated ent_pos_list

    """

    new_map = {}
    sents = []

    ent_pos_list = np.array(ent_pos_list)
    entity_start, entity_end = ent_pos_list[:, 0], ent_pos_list[:, 1] - 1
    # print('entity_start, entity_end', entity_start, entity_end)
    for i_t, token in enumerate(sent):
        tokens_wordpiece = tokenizer.tokenize(token)
        if add_marker:
            if i_t in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if i_t in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
        new_map[i_t] = len(sents)
        sents.extend(tokens_wordpiece)
    new_map[i_t + 1] = len(sents)

    entity_pos = [[new_map[s], new_map[e]] for s, e in ent_pos_list]

    if shift_right:
        entity_pos = np.array(entity_pos) + 1

    sents = sents[:max_seq_length - 2]

    input_ids = tokenizer.convert_tokens_to_ids(sents)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    return input_ids, entity_pos


# def preprocess_function(examples):
#     inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
#     targets = [ex[target_lang] for ex in examples["translation"]]
#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
#
#     # Setup the tokenizer for targets
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=max_target_length, truncation=True)
#
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


def fill_array_at_positions(length, positions, null_val=0, val=1, ):
    label_vec = [null_val] * length
    for pos in positions:
        label_vec[pos] = val
    return label_vec


def reverse_dict(dic, type="item2set"):
    new_dic = {}
    for token_pos, concept_ids in dic.items():
        for concept_id in concept_ids:
            if concept_id not in new_dic:
                new_dic[concept_id] = set()
            new_dic[concept_id].add(token_pos)
    return new_dic


def concat_steps(cur_steps, step_types, tokenizer=None, use_special_tag=True, is_target_text=False, use_eos=False, sep_token=None):
    pre_tokens = ''
    for j, step in enumerate(cur_steps):
        # start_pos += 1 + len(doc_list[j])
        # if j > 0: pre_tokens += " "
        pre_tokens += cur_steps[j].strip()
        # if j < len(concept_id_to_pos_list) - 1:
        if is_target_text:
            assert len(cur_steps) == len(step_types) + 1
            if j == len(cur_steps) - 1:
                break

        if use_special_tag == 1:
            if step_types[j] == "goal":
                pre_tokens += "[GOAL]"
                # pre_tokens += tokenizer.tokenize("[GOAL]")  # do not add SEP at end yet
                # assert pre_tokens[-1] == "[GOAL]"
            elif step_types[j] == "subgoal":
                pre_tokens += "[SUBGOAL]"
                # pre_tokens += tokenizer.tokenize("[SUBGOAL]")  # do not add SEP at end yet
                # assert pre_tokens[-1] == "[SUBGOAL]"
            elif step_types[j] == "event":
                pre_tokens += "[STEP]"  # do not add SEP at end yet
                # if j < len(cur_steps) - 1:
                #     pre_tokens += " [STEP]"  # do not add SEP at end yet
                # pre_tokens += " " + tokenizer.sep_token  # do not add SEP at end yet
            else:
                print("unknown steptype appeared")
                embed()
        elif use_special_tag == 0:
            # pre_tokens += "[STEP]"  # do not add SEP at end yet
            # if not use_eos:
            #     pre_tokens += tokenizer.sep_token  # " " +   # do not add SEP at end yet
            # else:
            #     pre_tokens += tokenizer.eos_token  # " " +   # do not add SEP at end yet
            # pre_tokens += tokenizer.sep_token
            if step_types[j] == "goal":
                pre_tokens += " [goal]. "
                # pre_tokens += tokenizer.tokenize("[GOAL]")  # do not add SEP at end yet
                # assert pre_tokens[-1] == "[GOAL]"
            elif step_types[j] == "subgoal":
                pre_tokens += " [subgoal]. "
                # pre_tokens += tokenizer.tokenize("[SUBGOAL]")  # do not add SEP at end yet
                # assert pre_tokens[-1] == "[SUBGOAL]"
            elif step_types[j] == "event":
                pre_tokens += " [step]. "  # do not add SEP at end yet
                # if j < len(cur_steps) - 1:
                #     pre_tokens += " [STEP]"  # do not add SEP at end yet
                # pre_tokens += " " + tokenizer.sep_token  # do not add SEP at end yet
            else:
                print("unknown steptype appeared")
                embed()

            # if j < len(cur_steps) - 1:
            # #     pre_tokens += " [STEP]"  # do not add SEP at end yet
            #     pre_tokens += tokenizer.sep_token# " " +   # do not add SEP at end yet
        elif use_special_tag == 2:
            pre_tokens += ". "
        elif use_special_tag == 3:
            pre_tokens += sep_token
    return pre_tokens


def convert_graph_data(cur_data):
    edge_index, edge_attr = torch.tensor(cur_data['edge_index'], dtype=torch.long), torch.tensor(cur_data['edge_attr'],
                                                                                                 dtype=torch.long)
    node_attr = torch.tensor(cur_data['edge_index'], dtype=torch.long)
    cur_g_data = Data(x=node_attr,
                      edge_index=edge_index,  # torch.tensor(edge_index, dtype=torch.long)
                      edge_attr=edge_attr)  # torch.tensor(, dtype=torch.long))  # .unsqueeze(1)

    return cur_g_data


def sents_to_token_ids_with_graph(list_of_sent=None, max_seq_length=None, tokenizer=None,
                                  add_sep_at_end=True, graphs=None, step_types=None, use_special_tag=True, max_node_num=200):
    # print("\n\nsents_to_token_ids_with_graph")
    """====update word indices after merging all steps===="""
    special_tks = ["[GOAL]", "[SUBGOAL]", "[STEP]", tokenizer.sep_token, "<ROOT>"]

    # g is like x, edge index, edge attr

    if not use_special_tag == 1:
        step_types_dict = {
            "goal": "[GOAL]",
            "subgoal": "[SUBGOAL]",
            "event": "[STEP]",
        }
    elif use_special_tag == 0:
        # step_types_dict = {k: tokenizer.sep_token for k, v in step_types_dict.items()}
        step_types_dict = {
            "goal": " [goal] ",
            "subgoal": " [subgoal] ",
            "event": " [step] ",
        }
    elif use_special_tag == 2:
        # step_types_dict = {k: tokenizer.sep_token for k, v in step_types_dict.items()}
        step_types_dict = {
            "goal": ".",
            "subgoal": ".",
            "event": ".",
        }

    merged_sent = [tokenizer.cls_token]
    start_pos = 1  # cls existss

    # print(graphs[0])
    # breakpoint()
    tmp_graphs = deepcopy(graphs)
    for i, g in enumerate(tmp_graphs):
        g['x'] = np.array(g['x'])
        # print("g['x'] original",i,  g['x'])
        # print("list_of_sent",i,  list_of_sent[i])
        if len(g['x']) and g['root'] != -1:  # empty graph
            g['x'] += start_pos
        # else:
        #     print("empty nodes")
        #     print(g)
        #     embed()
        merged_sent += (list_of_sent[i] + [step_types_dict[step_types[i]]])
        start_pos = len(merged_sent)
        # print(i, g['x'])
    # print('gs', tmp_graphs)
    # print('merged_sent', merged_sent)

    """====break words into tokens===="""

    new_map = {}
    original_pos_vec = []
    sents = []
    has_cls_at_start = merged_sent[0] == tokenizer.cls_token
    # print('has_cls_at_start', has_cls_at_start)
    for i_t, token in enumerate(merged_sent):
        token = token.strip()
        if not len(token):
            token = " "  # prevent empty token to disappear, making

        after_the_first_token = (has_cls_at_start and i_t > 1) or (not has_cls_at_start and i_t > 0)
        if token in special_tks or (after_the_first_token and merged_sent[
            i_t - 1].strip() in special_tks):  # for gpt like tokenizer which cares about space
            tokens_wordpiece = tokenizer.tokenize(token)
        else:
            tokens_wordpiece = tokenizer.tokenize((" " + token) if after_the_first_token else token)

        new_map[i_t] = len(sents)
        for _ in range(len(tokens_wordpiece)):
            original_pos_vec.append(i_t)
        sents.extend(tokens_wordpiece)
    new_map[i_t + 1] = len(sents)
    # print("\nsents", sents)
    # print("\nnew_map", new_map)

    """====Convert node ranges to new token indices===="""
    graph_instances = []
    root_indices = []
    out_of_range = False
    node_cnt = 0
    for i, g in enumerate(tmp_graphs):
        # print("graphs i", i)

        for j, (s, e) in enumerate(g['x']):
            # print("row j", j)

            if e >= max_seq_length - 1 or node_cnt == max_node_num:
                out_of_range = True
                break
            g['x'][j][0] = new_map[s]
            g['x'][j][1] = new_map[e]
            node_cnt += 1

        if out_of_range:
            break  # cut off the graphs which contain words that are out of range for lm limit
        if int(g['root']) < len(g['x']):
            cur_s, cur_e = g['x'][int(g['root'])]
            root_indices.extend(list(range(cur_s, cur_e)))
        else:
            print("root out of range", list_of_sent[i], "\n", g)
            # breakpoint()
        graph_instances.append(Data(x=get_tensor_long(g['x']), edge_index=get_tensor_long(g['edge_index']).t(),
                                    edge_attr=get_tensor_long(g['edge_attr'])))
    if not root_indices: root_indices = [0]
    # print("tmp_graphs",tmp_graphs)
    # print("root_indices",root_indices)
    # breakpoint()
    if not graph_instances:
        # print("\nnot graph_instances")
        graph_instances.append(
            Data(x=get_tensor_long([]), edge_index=get_tensor_long([]), edge_attr=get_tensor_long([])))
        print("empty gra")
        embed()
        # else:
    # for j, g in enumerate(graph_instances):
    #     for s,e in g.x.tolist():
    #         if s>=e:
    #             print(j, g.x)
    #             breakpoint()
    # print('\ngs', [g.x for g in graph_instances])

    # sents = sents[:max_seq_length - 2]
    sents = sents[:max_seq_length - 1]

    if add_sep_at_end:
        sents += [tokenizer.sep_token]
        original_pos_vec.append(original_pos_vec[-1] + 1)

    # print("final", sents)
    input_ids = tokenizer.convert_tokens_to_ids(sents)
    # print("sents",sents)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    return input_ids, new_map, token_type_ids, attention_mask, Batch.from_data_list(graph_instances), root_indices


class PrimitiveGenerationDataset(Dataset):

    def __init__(self, args, src_file, tokenizer=None, in_train=False):

        super().__init__()
        """========Init========="""

        self.tokenizer = tokenizer
        self.instances = []

        # Special Tokens
        self.SEP = tokenizer.sep_token_id
        self.CLS = tokenizer.cls_token_id
        self.BOS = tokenizer.bos_token_id
        self.EOS = tokenizer.eos_token_id
        self.SEP_TOKEN = tokenizer.sep_token
        self.CLS_TOKEN = tokenizer.cls_token
        self.BOS_TOKEN = tokenizer.bos_token
        self.EOS_TOKEN = tokenizer.eos_token

        if self.SEP is None:
            self.SEP = self.EOS
            self.SEP_TOKEN = self.EOS_TOKEN

        self.max_seq_len = args.max_seq_len
        # print("self.max_seq_len", self.max_seq_len)

        """========Load Cache========="""
        args.cache_filename = os.path.splitext(src_file)[0] + "_" + args.plm_class + "_" + args.data_mode + \
                              (f"_completion" if args.script_completion else "") + \
                              (f"_cgenerator" if args.pretrain_concept_generator else "") + \
                              (f"_primitive") + ".pkl"  #

        save_file = args.cache_filename
        print('\nReading data from {}.'.format(src_file))
        if os.path.exists(save_file) and args.use_cache:
            self.instances = load_file(save_file)
            print('load processed data from {}.'.format(save_file))
            return

        data_samples = load_file(src_file)
        skipped = 0

        print("restructured")

        for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
            if "src_text" not in sample:
                src_text, tgt_text = sample["text"], None
            else:
                src_text, tgt_text = sample["src_text"], sample["tgt_text"]
            model_inputs = self.tokenizer(src_text, padding=True, max_length=self.max_seq_len, truncation=True)

            if tgt_text is not None:
                with self.tokenizer.as_target_tokenizer():
                    # tgt_text = tgt_text.lower()
                    tgt = self.tokenizer(tgt_text, padding=True, max_length=self.max_seq_len,
                                         truncation=True)
                model_inputs["labels"] = tgt['input_ids']
                for key in ['input_ids', 'attention_mask']:
                    model_inputs["decoder_" + key] = tgt[key]
                model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

            # # don't do over long inputs
            exceed_max_len = False
            if max(len(tgt['input_ids']), len(model_inputs['input_ids'])) >= self.max_seq_len - 2:
                exceed_max_len = True
            if exceed_max_len:
                skipped += 1
                print("\n skipped", skipped)
                continue
            # if max(len(tgt['input_ids']),len( model_inputs['input_ids'])) >= self.max_seq_len - 2:
            #     skipped += 1
            #     print("skipped", skipped)
            #     continue

            self.instances.append({
                'tokenizer': tokenizer,
                'src_text': src_text,
                'tgt_text': tgt_text,
                "sample_id": i,
                'exceed_max_len': exceed_max_len,
            })
            self.instances[-1].update(model_inputs)

        """encode different parts into functions"""
        # save data
        if args.cache_filename:
            dump_file(self.instances, save_file)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance



def get_steps_text_from_ids(src_step_ids, step_db, add_finished=True, use_special_tag=False, tokenizer=None):
    # get from step database
    steps = [step_db[id] for id in src_step_ids]
    step_types = [step["step_type"] for step in steps]
    cur_steps = [step["step"] for step in steps]
    if add_finished:
        cur_steps += ["Finished"]
    tgt_text = concat_steps(cur_steps, step_types, tokenizer, use_special_tag=use_special_tag,
                            is_target_text=True, sep_token=". ")
    return tgt_text


def get_pseudo_target(neg_samples_dict, tgt_steps, tgt_step_types, args, tokenizer):
    sample3 = None
    paraphrase_locs = neg_samples_dict["paraphrase_locs"]
    paraphrase_steps = [st.rstrip(".") for st in neg_samples_dict["paraphrase_steps"]]
    if paraphrase_locs:
        tmp_tgt_steps = []
        tmp_tgt_step_types = []
        cur_spot = 0
        for k, cur_tgt_step in enumerate(tgt_steps[:-1] if args.add_finished else tgt_steps):
            tmp_tgt_steps.append(cur_tgt_step)
            tmp_tgt_step_types.append(tgt_step_types[k])
            while cur_spot < len(paraphrase_locs) and k == int(paraphrase_locs[cur_spot][1]):
                tmp_tgt_steps.append(paraphrase_steps[cur_spot])
                tmp_tgt_step_types.append("event")
                cur_spot += 1
        if args.add_finished:
            tmp_tgt_steps += ["Finished"]
        sample3 = concat_steps(tmp_tgt_steps, tmp_tgt_step_types, tokenizer, use_special_tag=args.use_special_tag,
                               is_target_text=True, sep_token=". ")
    return sample3


def get_neg_samples(sample_types, neg_samples_dict, tgt_text, tgt_steps, tgt_step_types, step_db, args, tokenizer, cl_empty_mode):
    num_types_list = [sample_types.count(str(num)) for num in range(4)]
    res = []
    backups = []
    # loop_vs=["","v2","v3","v4"]

    neg_samples_dicts = [neg_samples_dict, neg_samples_dict["v2"], neg_samples_dict["v3"], neg_samples_dict["v4"]] #, neg_samples_dict["v3"], neg_samples_dict["v4"]

    for i, num_type in enumerate(num_types_list):
        if i == 0:
            pass
            # # shuffle
            # shuffled_tgt_step_ids = neg_samples_dict_v["shuffled_tgt_step_ids"]
            # if shuffled_tgt_step_ids:
            #     sample1 = get_steps_text_from_ids(shuffled_tgt_step_ids, step_db, add_finished=True,
            #                                       use_special_tag=args.use_special_tag, tokenizer=tokenizer)
        elif i == 1:
            for vs in neg_samples_dicts[1:][:num_type]:
                # entity_swaps
                entity_swaps = vs["entity_swaps"]
                sample2 = None
                if entity_swaps:
                    tmp_tgt_text = tgt_text
                    for src_entity, tgt_entity in entity_swaps:
                        tmp_tgt_text = tmp_tgt_text.replace(src_entity, tgt_entity)
                    sample2 = tmp_tgt_text
                res.append(sample2)
                cur_replace = None
                if sample2 is None and cl_empty_mode==0:
                    cur_replace = get_steps_text_from_ids(vs["rand_tgts_from_db"][1], step_db, add_finished=True,
                                                          use_special_tag=args.use_special_tag, tokenizer=tokenizer)
                backups.append(cur_replace)
        elif i == 2:
            for vs in neg_samples_dicts[:num_type]:
                # paraphrase_locs
                sample3 = get_pseudo_target(vs, tgt_steps, tgt_step_types, args, tokenizer)
                res.append(sample3)
                cur_replace = None
                if sample3 is None and cl_empty_mode==0:
                    cur_replace = get_steps_text_from_ids(vs["rand_tgts_from_db"][2], step_db, add_finished=True,
                                                          use_special_tag=args.use_special_tag, tokenizer=tokenizer)
                backups.append(cur_replace)
        elif i == 3:
            for vs in neg_samples_dicts[:num_type]:
                rand_tgts_from_db = vs["rand_tgts_from_db"]
                sample4 = get_steps_text_from_ids(rand_tgts_from_db[3], step_db, add_finished=True,
                                                  use_special_tag=args.use_special_tag, tokenizer=tokenizer)
                res.append(sample4)
                backups.append(None)

    # sample4_tmp = get_steps_text_from_ids(neg_samples_dict_tmp["rand_tgts_from_db"][3], step_db, add_finished=True,
    #                                   use_special_tag=args.use_special_tag, tokenizer=tokenizer)
    # assert sample4_tmp==sample4

    return res, backups


class WikiHowDatasetEST(Dataset):

    def __init__(self, args, src_file, tokenizer=None, tcg_file=None, generation_mode=False, step_db_file=None, in_train=False, factors_file=None):

        super().__init__()
        """========Init========="""

        self.tokenizer = tokenizer
        self.instances = []

        # Special Tokens
        self.SEP = tokenizer.sep_token_id
        self.CLS = tokenizer.cls_token_id
        self.BOS = tokenizer.bos_token_id
        self.EOS = tokenizer.eos_token_id
        self.SEP_TOKEN = tokenizer.sep_token
        self.CLS_TOKEN = tokenizer.cls_token
        self.BOS_TOKEN = tokenizer.bos_token
        self.EOS_TOKEN = tokenizer.eos_token
        sp = spacy.load('en_core_web_sm')
        all_stopwords = sp.Defaults.stop_words
        all_stopwords.update(set(stopwords.words('english')))
        all_stopwords.update(set(STOPWORDS))
        lemmatizer = WordNetLemmatizer()

        if self.SEP is None:
            self.SEP = self.EOS
            self.SEP_TOKEN = self.EOS_TOKEN

        self.max_seq_len = args.max_seq_len
        # print("self.max_seq_len", self.max_seq_len)

        """========Load Cache========="""
                              # (f"_augment" if args.augment_data else "") + \
                              # (f"_hie" if args.hierachy else "_nohie") + \
                              # (f"_silvereval" if args.silver_eval else "") + \
        args.cache_filename = os.path.splitext(src_file)[0] + "_" + args.plm_class + \
                              (f"_compl" if args.script_completion else "") + \
                              (f"_intrain" if in_train else "") + \
                              (f"_cgen" if args.pretrain_concept_generator else "") + \
                              (f"_upc" if args.use_pretrained_concepts else "") + \
                              (f"_egold" if args.eval_gold_concepts else "") + \
                              (f"_debug" if args.debug else "") + \
                              (f"_ugf" if args.use_generated_factors else "") + \
                              (f"_tgold" if args.train_gold_concepts else "") + \
                              (f"_nts{args.num_tgt_steps}" if args.num_tgt_steps != -1 and args.script_completion else "") + \
                              (f"_sptag{args.use_special_tag}") + (f"_{args.data_mode}") + \
                              (f"_factorexp" if args.factor_expander else "") + \
                              (f"_specialp" if args.special_prompt else "") + \
                              (f"_add_fin{args.add_finished}") + \
                              (f"_topkp{args.topk_prompt}") + \
                              (f"_clsamp{args.cl_sample_types}") + \
                              (f"_clem{args.cl_empty_mode}") + \
                              (f"_fewk{args.fill_empty_with_cg}") + \
                              (f"_ncm{args.nb_collect_mode}") + \
                              (f"_rankmode{args.rank_mode}") + \
                              (f"_hl{args.hist_length}") + \
                              (f"_ltgt" if args.lower_case_tgt else "") + (f"_ttype" if args.has_tgt_type else "") + \
                              "_" + args.task_mode + ".pkl"  #
        # (("_"+args.data_mode) if is_train else "") +
        #                       (f"_{args.fs_mode}") + \
        save_file = args.cache_filename
        print('\nReading data from {}.'.format(src_file))
        if os.path.exists(save_file) and args.use_cache:
            self.instances = load_file(save_file)
            print('load processed data from {}.'.format(save_file))
            return

        """========Load File========="""
        grounded = load_file(src_file)
        # if args.debug: grounded = grounded[:8]

        """========Special PLM Tokenization========="""
        is_t5 = "t5" in args.plm
        if args.use_special_tag != 1:
            special_token_ids = {self.SEP, }
            special_tokens = {self.SEP_TOKEN}
        else:
            special_token_ids = {self.SEP,
                                 tokenizer.convert_tokens_to_ids(["[GOAL]"])[0],
                                 tokenizer.convert_tokens_to_ids(["[SUBGOAL]"])[0],
                                 tokenizer.convert_tokens_to_ids(["[STEP]"])[0],
                                 tokenizer.convert_tokens_to_ids(["<ROOT>"])[0],
                                 }
            special_tokens = {self.SEP_TOKEN, "[GOAL]", "[SUBGOAL]", "[STEP]", "<ROOT>"}  #

        step_types_dict = {
            "goal": "[GOAL]",
            "subgoal": "[SUBGOAL]",
            "event": "[STEP]",
        }
        if args.use_special_tag == 0:
            step_types_dict = {
                "goal": "[goal].",
                "subgoal": "[subgoal].",
                "event": "[step].",
            }
        if args.use_special_tag == 2:
            step_types_dict = {
                "goal": ". ",
                "subgoal": ". ",
                "event": ". ",
            }
        if args.use_special_tag == 3:
            step_types_dict = {
                "goal": self.SEP_TOKEN,
                "subgoal": self.SEP_TOKEN,
                "event": self.SEP_TOKEN,
            }

        mask_token_id = tokenizer.mask_token_id
        has_cls = not is_t5
        print("\nSpecial_token_ids", special_token_ids)

        # step_db = load_file(step_db_file)
        step_dbs = [load_file(f) for f in [f"{NS_ss9_dir}separated/data_{split}.json" for split in ['train', 'dev', 'test']]]

        candidate_pool = {}
        augmented_data = []
        ## create augmented data
        mm = 0
        skipped = 0

        cur_valid_index = 0
        if args.use_generated_factors:
            assert not args.pretrain_concept_generator
            saved_factors = load_file(tcg_file)

            if in_train and args.train_gold_concepts and args.unordered_gold_concepts:
                factor_key_name = "all_factors"
            else:
                factor_key_name = "tgt_text" if in_train and args.train_gold_concepts else 'predicted'

            # unordered_gold_concepts

            if in_train and args.train_gold_concepts and args.unordered_gold_concepts:
                factor_db = {}
                for item in saved_factors:
                    if 'predicted' in item:
                        allf = deepcopy(item[factor_key_name])
                        np.random.shuffle(allf)
                        factor_db[(item["global_doc_id"], item["doc_id"], item["path_id"])] = "[" + ", ".join(allf) + "]"
            else:
                factor_db = {(item["global_doc_id"], item["doc_id"], item["path_id"]): item[factor_key_name] for item in saved_factors if 'predicted' in item}

            print("len(factor_db)", len(factor_db))
            print("len(saved_factors)", len(saved_factors))
        visited_sample_ids = set()
        if False:  # args.pretrain_concept_generator and in_train

            pass
        else:
            cnt_err = 0
            records = []

            # tmp_tr=load_file("data/wikihow/subset9/grounded/full_data_train.grounded_with_negsamples.json")
            for i, path in enumerate(tqdm(grounded)):  # this indicates ith path after breaking out all articles into individual paths

                # if args.debug and i>30: break
                # if i not in [490, 491]: continue

                # print("\n\n\nsample",i)
                global_doc_id = path["global_doc_id"]
                doc_id = path["doc_id"]
                path_id = path["path_id"]
                if (global_doc_id, doc_id, path_id) == (17894, 8851, 0): continue
                src_step_ids = path["src_step_ids"]
                if in_train and (global_doc_id, doc_id, path_id, len(src_step_ids)) in visited_sample_ids: continue
                visited_sample_ids.add((global_doc_id, doc_id, path_id, len(src_step_ids)))

                doc_type = path["doc_type"]
                src_step_ids = path["src_step_ids"]
                tgt_step_ids = path["tgt_step_ids"]

                step_db = step_dbs[path['srcsplit_id']]
                if len(src_step_ids) > 20:
                    continue

                if args.hist_length and len(src_step_ids) > int(args.hist_length)+1: #+1 for title
                    continue
                if args.hierachy == 0:
                    src_step_ids = clear_subgoals3(src_step_ids, step_db)
                    tgt_step_ids = clear_subgoals3(tgt_step_ids, step_db)

                if args.num_tgt_steps != -1 and args.script_completion and len(tgt_step_ids) > args.num_tgt_steps:
                    continue

                categories = path["categories"]
                if args.data_mode != "full" and args.abbr2category[args.data_mode] not in categories:
                    continue

                # entities = path['entities']
                # list_of_triggers = path['triggers']

                # get from step database
                steps = [step_db[id] for id in src_step_ids]
                step_types = [step["step_type"] for step in steps]

                cur_steps = [step["step"] for step in steps]
                tokens = [step["tokens"] for step in steps]
                subgoal = path["subgoal_text"] if doc_type == "methods" else ""
                if subgoal:
                    tmp = deepcopy(cur_steps)
                    tmp[0] += " (" + subgoal + ")"
                    cur_steps = tmp

                    tmp = deepcopy(tokens)
                    # tmp[0].extend(all_factors_tokens)
                    tmp[0] += [" (" + subgoal + ")"]
                    tokens = tmp

                # amr_graphs = [step["amr"]["graph"] for step in steps]
                # cur_poss = [step["token_features"] for step in steps]

                # tgt
                raw_tgt_steps = [step_db[id] for id in tgt_step_ids]
                tgt_steps = [step_db[id]["step"].lower() if args.lower_case_tgt else step_db[id]["step"] for id in tgt_step_ids]  # .lower()
                tgt_step_types = [step_db[id]["step_type"] for id in tgt_step_ids]
                if args.has_tgt_type:
                    tmp = []
                    for j, step_type in enumerate(tgt_step_types):
                        tmp.append(tgt_steps[j] + " " + step_types_dict[step_type].lower())
                    tgt_steps = tmp
                if args.add_finished:
                    tgt_steps += ["Finished"]

                all_steps = steps + raw_tgt_steps

                # don't do over long inputs
                tmp_tok_lens1 = len(self.tokenizer(self.SEP_TOKEN.join(cur_steps), padding=True, max_length=self.max_seq_len,
                                                   truncation=True)['input_ids'])
                tmp_tok_lens2 = len(self.tokenizer(", ".join(tgt_steps), padding=True, max_length=self.max_seq_len,
                                                   truncation=True)['input_ids'])
                if max(tmp_tok_lens1, tmp_tok_lens2) >= self.max_seq_len - 2:
                    skipped += 1
                    # print("skipped")
                    # print("\n\n\nsample", tmp_tok_lens1, tmp_tok_lens2)
                    continue

                if "gpt2" in args.plm and in_train:
                    cur_steps += tgt_steps
                    step_types += tgt_step_types

                """=============add factor prompt============="""

                # def is_candidate_factor(tok, feat):
                #     return "NN" in feat and tok not in all_stopwords
                #
                # all_factors = OrderedDict()
                # buffer = []
                # cluster = defaultdict(list)
                # cluster2 = defaultdict(list)
                # cw2hw = {}
                # for k, step in enumerate(all_steps):
                #     for j, (tok, feat) in enumerate(step["token_features"]):
                #         if is_candidate_factor(tok, feat):
                #             buffer.append(tok)
                #             if j == len(step["token_features"]) - 1 or not is_candidate_factor(*step["token_features"][j + 1]):  # headword
                #
                #                 complete_word = " ".join(buffer)
                #                 if k > 0 and complete_word not in all_factors: all_factors[complete_word] = 1
                #                 cluster[tok.lower()].append((k, j))
                #                 cluster2["_".join(buffer).lower()].append((k, j))
                #                 buffer.clear()
                # if buffer:
                #     complete_word = " ".join(buffer)
                #     if complete_word not in all_factors: all_factors[complete_word] = 1
                #
                # all_factors = list(all_factors.keys())
                # all_factors_text = "[" + ", ".join(all_factors) + "]"
                # all_factors_tokens = ["["]
                # for j, factor in enumerate(all_factors):
                #     all_factors_tokens.append(factor)
                #     if j != len(all_factors) - 1:
                #         all_factors_tokens.append(",")
                # all_factors_tokens.append("]")

                if args.use_generated_factors:
                    if args.use_pretrained_concepts:
                        all_factors_text = factor_db.get((global_doc_id, doc_id, path_id), "")
                        # print(all_factors_text)
                        if not all_factors_text.strip():
                            # all_factors_text='[]'
                            breakpoint()
                    elif args.silver_eval:
                        neighbor_factor_txt = path["neighbor_factor_txt"]
                        # neighbor_goal = path["neighbor_goal"]
                        gold_factor_txt = path["gold_factor_txt"]
                        to_use_gold_concepts = (in_train and args.train_gold_concepts) or (not in_train and args.eval_gold_concepts)
                        all_factors_text = gold_factor_txt if to_use_gold_concepts else neighbor_factor_txt
                    else:
                        tk_name = f"top_{args.topk_prompt}"
                        tmp_path = path[args.rank_mode][tk_name] if len(args.rank_mode.strip()) else path[tk_name]
                        neighbor_factor_txt = tmp_path["neighbor_factor_txt"] if args.nb_collect_mode == 0 else tmp_path["union_neighbor_factor_txt"]
                        if args.gold_percent:
                            neighbor_factor_txt = path[args.rank_mode][f"neighbor_factor_txt_gold{args.gold_percent}"]

                        # neighbor_goal = path["neighbor_goal"]
                        # if neighbor_factor_txt.strip() == "[]" or not neighbor_factor_txt.strip():
                        #     breakpoint()
                        if args.fill_empty_with_cg and (neighbor_factor_txt.strip() == "[]" or not neighbor_factor_txt.strip()):
                            neighbor_factor_txt = path[args.rank_mode]['cg_neighbor_factor_txt']
                        gold_factor_txt = path[args.rank_mode]["top_1"]["gold_factor_txt"]
                        to_use_gold_concepts = (in_train and args.train_gold_concepts) or (not in_train and args.eval_gold_concepts)
                        all_factors_text = gold_factor_txt if to_use_gold_concepts else neighbor_factor_txt
                    if args.special_prompt:
                        if all_factors_text.strip():
                            assert all_factors_text[0] == "[" and all_factors_text[-1] == "]"
                            middle_text = all_factors_text[1:-1]
                            if not middle_text: middle_text = " "
                            all_factors_text = "###" + middle_text + "###"

                    cur_valid_index += 1

                if args.factor_expander and not args.prompt_at_back:
                    tmp = deepcopy(cur_steps)
                    if not args.prompt_at_back:
                        tmp[0] += " " + all_factors_text
                    else:
                        pass
                        # tmp[-1] += " " + all_factors_text

                    cur_steps = tmp

                    tmp = deepcopy(tokens)
                    # tmp[0].extend(all_factors_tokens)
                    tmp[0].append(all_factors_text)
                    tokens = tmp

                # tgt_step_types += [step_types_dict["event"]]

                """=============augment data============="""
                if args.augment_data and in_train:
                    # self.augmented_instances=[]
                    replacement = {}
                    # all_headword_factors = [ f.split()[-1] for f in all_factors]
                    # for f in all_headword_factors:
                    #     lemm = lemmatizer.lemmatize(f)
                    #     if f not in replacement:
                    #         replacement[f] = get_candidate_pool2(lemm, candidate_pool)
                    # for r in replacement:
                    #     if r not in all_headword_factors:
                    #         replacement[r] = []
                    all_steps_tokens = [step["tokens"] for step in all_steps]
                    tmp = deepcopy(all_steps_tokens)
                    random_size = 5
                    for lemma in cluster:
                        if lemma not in replacement:
                            candidates = get_candidate_pool2(lemma, candidate_pool, return_random=True, random_size=random_size)
                            if len(candidates): replacement[lemma] = candidates
                    for pos in range(random_size):  # how many new samples to generate for each samples
                        for lemma, candidates in replacement.items():  # candidates mean new terms pool
                            for dex_id, (step_id, token_id) in enumerate(cluster[lemma]):
                                tmp[step_id][token_id] = candidates[min(pos, len(candidates) - 1)]

                        tmp_src_tokens, tmp_tgt_tokens = tmp[:len(steps)], tmp[len(steps):]
                        tmp_src_steps = [" ".join(l) for l in tmp_src_tokens]
                        merged_sent, start_positions = merge_all_sents(tmp_src_tokens, tokenizer=tokenizer, step_types_dict=step_types_dict,
                                                                       step_types=step_types, has_cls=has_cls)
                        input_ids, new_map, _, _, attention_mask, subtokens = sent_to_token_ids(merged_sent, self.max_seq_len, tokenizer,
                                                                                                shift_right=False, add_sep_at_end=args.use_special_tag != 3,
                                                                                                has_cls_at_start=has_cls, special_tks=special_tokens,
                                                                                                end_token=self.SEP_TOKEN)
                        model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                        tmp_tgt_steps = [" ".join(l) for l in tmp_tgt_tokens]
                        tmp_tgt_steps = [s.lower() if args.lower_case_tgt else s for s in tmp_tgt_steps] + (["Finished"] if args.add_finished else [])
                        if args.script_completion:
                            tgt_text = concat_steps(tmp_tgt_steps, tgt_step_types, tokenizer, use_special_tag=args.use_special_tag,
                                                    is_target_text=True, use_eos=is_t5, sep_token=". ")
                        else:
                            tgt_text = tgt_steps[0]
                        with self.tokenizer.as_target_tokenizer():
                            # tgt_text = tgt_text.lower()
                            tgt = self.tokenizer(tgt_text, padding=True, max_length=self.max_seq_len,
                                                 truncation=True)
                        model_inputs["labels"] = tgt['input_ids']
                        for key in ['input_ids', 'attention_mask']:
                            model_inputs["decoder_" + key] = tgt[key]
                        model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

                        # if max(len(input_ids), len(tgt['input_ids'])) >= self.max_seq_len - 2:
                        #     skipped += 1
                        #     continue
                        self.instances.append({
                            'tokenizer': tokenizer,
                            'global_doc_id': global_doc_id,
                            'doc_id': doc_id,
                            'path_id': path_id,
                            'doc_type': doc_type,
                            'src_text': tmp_src_steps,
                            'input_text': self.SEP_TOKEN.join(tmp_src_steps),
                            'tgt_text': tgt_text,
                            "sample_id": i,
                            'src_step_ids': src_step_ids,
                            'tgt_step_ids': tgt_step_ids,
                            "categories": categories,
                            "synthetic": 1,
                        })
                        self.instances[-1].update(model_inputs)
                    continue

                "=====Debug====="""
                assert len(step_types) == len(cur_steps) or "gpt" in args.plm, print("step_types", step_types, '\ncur_steps', cur_steps)

                """============MODEL INPUT""============"""
                pre_tokens = concat_steps(cur_steps, step_types, tokenizer, use_special_tag=args.use_special_tag, use_eos=is_t5, sep_token=self.SEP_TOKEN)
                if args.factor_expander and args.prompt_at_back:
                    pre_tokens += all_factors_text + self.SEP_TOKEN
                """====aggregate factor positions===="""
                # tokenization for PLM
                merged_sent, start_positions = merge_all_sents(tokens, tokenizer=tokenizer, step_types_dict=step_types_dict, step_types=step_types, has_cls=has_cls)

                if args.factor_expander and args.prompt_at_back:
                    merged_sent += [all_factors_text + self.SEP_TOKEN]

                input_ids, new_map, _, _, attention_mask, subtokens = sent_to_token_ids(merged_sent, self.max_seq_len, tokenizer, shift_right=False,
                                                                                        add_sep_at_end=args.use_special_tag != 3, has_cls_at_start=has_cls,
                                                                                        special_tks=special_tokens,
                                                                                        end_token=self.SEP_TOKEN)

                # if tokenizer.batch_decode([input_ids])[0].split("</s>")[0].count("[") >= 2 or tokenizer.batch_decode([input_ids])[0].split("</s>")[0].count("]") >= 2:
                #     print(tokenizer.batch_decode([input_ids]))
                # breakpoint()

                token_tags = [0] * len(input_ids)
                # for j, s in enumerate(cur_poss):
                #     # if step_types[j] != "step":
                #     #     continue
                #     for k, (_, pos_tag) in enumerate(s):
                #         # print(_, pos_tag)
                #         if "VB" in pos_tag:
                #             token_pos_in_merged_sent = start_positions[j] + k
                #             for p in range(new_map[token_pos_in_merged_sent], new_map[token_pos_in_merged_sent+1]):
                #                 if p<len(token_tags):
                #                     token_tags[p] = 1

                if not is_t5:  # and not args.factor_expander
                    # update token ranges for entities/triggers
                    # entities["<<triggers>>"] = list_of_triggers
                    # ranges_batch = exclude_columns_for_lists(list(entities.values()), [1])  # has num_ents lists
                    # ranges_batch = accumulate_ranges(ranges_batch, start_positions)
                    # ranges_batch, _ = update_ranges_with_subtoken_mapping(ranges_batch, new_map, self.max_seq_len, filter_oor=True)
                    # idxs, masks, max_token_num, max_token_len = token_lens_to_idxs(ranges_batch)
                    # entities = {
                    #     "ranges": ranges_batch,
                    #     "idxs": get_tensor_long([[id for b in idxs for id in b]]),
                    #     "masks": get_tensor_float([[m for b in masks for m in b]]),
                    #     "max_token_num": max_token_num * len(ranges_batch),  # we treat each entity as a batch, we gather from the token embeddings from this sample
                    #     "max_token_len": max_token_len,
                    #     "num_mentions": [len(r) for r in ranges_batch],
                    #     "max_mention_num": max_token_num,
                    #     "num_entities": len(ranges_batch),  # we treat each entity as a batch, we gather from the token embeddings from this sample
                    # }
                    # pad_mask = []
                    # for mention_num in entities["num_mentions"]:
                    #     pad_mask.extend([1] * mention_num + [0] * (entities["max_mention_num"] - mention_num))
                    # entities["pad_mask"] = pad_mask  # for padded mentions
                    #
                    # # aggregating small amr graphs
                    # ranges_batch = [[[j] + span_range for span_range in graph['token_span']] for j, graph in enumerate(amr_graphs)]
                    # ranges_batch = accumulate_ranges(ranges_batch, start_positions)
                    # node_span_batch, valid_node_masks = update_ranges_with_subtoken_mapping(ranges_batch, new_map, self.max_seq_len)
                    # # node_span_batch=[[start, end] for j, node_spans in enumerate(node_span_batch) for _, start, end in node_spans]
                    # text_graph = {
                    #     "graph": aggregate_graphs(amr_graphs, node_span_batch, valid_node_masks, args.max_node_num),
                    # }

                    # if args.debug:
                    #     if i < 80 or i >180:
                    #         continue
                    #     if i==80:
                    #         breakpoint()
                    #         embed()
                    if "gpt2" in args.plm:
                        if in_train: pre_tokens += " " + tokenizer.eos_token
                        inputss = tokenizer(pre_tokens, padding=True, max_length=self.max_seq_len - 1, truncation=True)
                        input_ids, attention_mask = inputss["input_ids"], inputss["attention_mask"]
                    model_inputs = {'input_ids': input_ids,
                                    'attention_mask': attention_mask,
                                    # 'entities': entities,
                                    # 'text_graph': text_graph,
                                    # [ent_range_dict[ent] for ent in ent_range_dict],#ent_range_dict,#[ent_range_dict["verb_ranges"] for ent in ent_range_dict],
                                    }
                    model_inputs["token_type_ids"] = [0] * len(model_inputs["input_ids"])
                else:
                    model_inputs = {'input_ids': input_ids,
                                    'attention_mask': attention_mask,
                                    }
                model_inputs['token_tags'] = token_tags

                """====Decoder Inputs===="""
                #
                if args.script_completion:
                    if "gpt2" in args.plm and in_train:
                        tgt_text = pre_tokens
                    else:
                        tgt_text = concat_steps(tgt_steps, tgt_step_types, tokenizer, use_special_tag=args.use_special_tag, is_target_text=True, use_eos=is_t5, sep_token=". ")
                    # print("\ntgt_text\t", tgt_text)
                    # with self.tokenizer.as_target_tokenizer():
                    #     tgts = self.tokenizer(tgt_text, padding=True, max_length=self.max_seq_len,
                    #                          truncation=True)
                    # model_inputs["complete_labels"] = tgts['input_ids']  # [1:]
                    # for key in ['input_ids', 'attention_mask']:
                    #     model_inputs["complete_decoder_" + key] = tgts[key]
                    # model_inputs["complete_decoder_" + "token_type_ids"] = [0] * len(tgts["input_ids"])
                else:
                    tgt_text = tgt_steps[0]

                # if args.pretrain_concept_generator and not in_train:
                #     tgt_text = all_factors_text
                #     src_tokens = self.tokenizer(cur_steps[0], padding=True, max_length=self.max_seq_len - 1,
                #                                 truncation=True)
                #     model_inputs["input_ids"] = src_tokens["input_ids"]
                #     model_inputs["attention_mask"] = src_tokens["attention_mask"]
                #     # cur_valid_index+=1
                if args.cl_exp_type == "a":
                    args.ct = ""
                # print(args.ct)
                # print(args.cl_empty_mode)
                # print(args.fill_empty_neg_with_null_str)
                # print(args.cl_sample_types)

                if False:
                    with self.tokenizer.as_target_tokenizer():
                        tgt = self.tokenizer(tgt_text, padding=True, max_length=self.max_seq_len - 1,
                                             truncation=True)
                        # print("tgt",tgt)
                        tgt['input_ids'] = [tokenizer.pad_token_id] + tgt['input_ids']
                        tgt['attention_mask'] = [1] + tgt['attention_mask']
                else:
                    with self.tokenizer.as_target_tokenizer():
                        # tgt_text = tgt_text.lower()

                        """get negative samples"""
                        negative_samples = []
                        if in_train:
                            neg_samples_dict = path['neg_samples_dict']

                            if not args.orig_cl_way:
                                negative_samples, backups=get_neg_samples(args.cl_sample_types, neg_samples_dict, tgt_text, tgt_steps, tgt_step_types, step_db, args, tokenizer, cl_empty_mode=args.cl_empty_mode)

                            else:
                                """ORIG CODE START HERE"""

                                neg_samples_dict_v = neg_samples_dict  # adjust
                                if args.ct == "v2":
                                    neg_samples_dict_v = neg_samples_dict["v2"]

                                sample1, sample2, sample3, sample4 = None, None, None, None

                                # shuffle
                                shuffled_tgt_step_ids = neg_samples_dict_v["shuffled_tgt_step_ids"]
                                if shuffled_tgt_step_ids:
                                    sample1 = get_steps_text_from_ids(shuffled_tgt_step_ids, step_db, add_finished=True,
                                                                      use_special_tag=args.use_special_tag, tokenizer=tokenizer)

                                # entity_swaps
                                entity_swaps = neg_samples_dict_v["entity_swaps"]
                                if entity_swaps:
                                    tmp_tgt_text = tgt_text
                                    for src_entity, tgt_entity in entity_swaps:
                                        tmp_tgt_text = tmp_tgt_text.replace(src_entity, tgt_entity)
                                    sample2 = tmp_tgt_text
                                    if args.cl_exact_location and "1" not in args.cl_sample_types: sample2=""

                                sample2_multi = None
                                entity_swapss = neg_samples_dict["v2"]["entity_swaps"]
                                if entity_swapss:
                                    tmp_tgt_text2 = tgt_text
                                    for src_entity, tgt_entity in entity_swapss:
                                        tmp_tgt_text2 = tmp_tgt_text2.replace(src_entity, tgt_entity)
                                    sample2_multi = tmp_tgt_text2

                                # if i==42368:
                                #     embed()

                                # paraphrase_locs
                                sample3 = get_pseudo_target(neg_samples_dict_v, tgt_steps, tgt_step_types, args, tokenizer)
                                sample3_multi = get_pseudo_target(neg_samples_dict["v2"], tgt_steps, tgt_step_types, args, tokenizer) if "v2" in neg_samples_dict else None
                                if args.cl_exact_location and "2" not in  args.cl_sample_types and sample3 is not None: sample3 = ""


                                rand_tgts_from_db = neg_samples_dict["rand_tgts_from_db"]

                                # empty cases
                                negative_samples = [sample1, sample2, sample3]
                                # if args.cl_empty_mode != 0:
                                sample4 = get_steps_text_from_ids(rand_tgts_from_db[3], step_db, add_finished=True,
                                                                  use_special_tag=args.use_special_tag, tokenizer=tokenizer)
                                sample4_multi = get_steps_text_from_ids(neg_samples_dict["v2"]["rand_tgts_from_db"][3], step_db, add_finished=True,
                                                                  use_special_tag=args.use_special_tag, tokenizer=tokenizer)

                                negative_samples.append(sample4)

                            """ORIG CODE END HERE"""

                            negative_samples_new = []
                            for k, samp in enumerate(negative_samples):
                                if samp is None:
                                    if args.cl_empty_mode == 0:
                                        if args.orig_cl_way:
                                            samp = get_steps_text_from_ids(rand_tgts_from_db[k], step_db, add_finished=True,
                                                                           use_special_tag=args.use_special_tag, tokenizer=tokenizer)

                                        else:
                                            if backups[k] is None:
                                                breakpoint()
                                            samp=backups[k]
                                    elif args.cl_empty_mode == 1:
                                        samp = ""

                                # print(k, samp)
                                tokenized = self.tokenizer(samp, padding=True, max_length=self.max_seq_len,
                                                           truncation=True)
                                tokenized = modify_dict_keys(tokenized, prefix="decoder_")
                                tokenized["decoder_" + "token_type_ids"] = [0] * len(tokenized["decoder_input_ids"])
                                # labels
                                tokenized["is_empty"] = False
                                if args.cl_empty_mode == 1 and not len(samp) and not args.fill_empty_neg_with_null_str:
                                    tokenized["is_empty"] = True
                                # is None

                                if "gpt2" not in args.plm or in_train:
                                    tokenized["labels"] = tokenized['decoder_input_ids']
                                else:
                                    assert False

                                negative_samples_new.append(tokenized)
                            negative_samples = negative_samples_new

                        tgt = self.tokenizer(tgt_text, padding=True, max_length=self.max_seq_len,
                                             truncation=True)

                    if "gpt2" not in args.plm or in_train:
                        model_inputs["labels"] = tgt['input_ids']
                    else:
                        model_inputs["labels"] = input_ids

                    # if not is_t5:
                    for key in ['input_ids', 'attention_mask']:
                        model_inputs["decoder_" + key] = tgt[key]
                    model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

                """====append instances ===="""
                if args.task_mode in ["gen", "ret", "prt"]:
                    ## output
                    # a giant graph
                    # node ranges
                    # entity/trigger each with global ranges
                    # tmp_mdi=deepcopy(model_inputs)
                    # for kk in tmp_mdi:
                    #     if not isinstance(tmp_mdi[kk], list):
                    #         tmp_mdi[kk]=tmp_mdi[kk].tolist()
                    #
                    # tmp_ng=deepcopy(negative_samples)
                    # for ng in tmp_ng:
                    #     for kk in ng:
                    #         if not isinstance(ng[kk], list) and kk!="is_empty":
                    #             ng[kk]=ng[kk].tolist()
                    # tmp_mdi["tmp_ng"]=tmp_ng
                    # tmp_mdi["pos"]=i
                    # records.append(tmp_mdi)
                    # model_inputs = {'input_ids': input_ids,
                    #                 'attention_mask': attention_mask,
                    #                 # 'entities': entities,
                    #                 # 'text_graph': text_graph,
                    #                 # [ent_range_dict[ent] for ent in ent_range_dict],#ent_range_dict,#[ent_range_dict["verb_ranges"] for ent in ent_range_dict],
                    #                 }

                    self.instances.append({
                        'tokenizer': tokenizer,
                        'global_doc_id': global_doc_id,
                        'doc_id': doc_id,
                        'path_id': path_id,
                        'doc_type': doc_type,
                        'src_text': pre_tokens,  # cur_steps,
                        'input_text': pre_tokens,
                        'input_text_from_ids': tokenizer.batch_decode([input_ids], skip_special_tokens=False)[0],
                        'tgt_text': tgt_text,
                        "sample_id": i,
                        'src_step_ids': src_step_ids,
                        'tgt_step_ids': tgt_step_ids,
                        "categories": categories,
                        "negative_samples": negative_samples,
                        "in_train": in_train
                        # "history_length":history_length
                        # 'nbs_src_text':nbs_src_text,
                        # "concept_set": concept_set
                    })

                    self.instances[-1].update(model_inputs)

        def clean_sents(sents):
            return [sent[:-1] if sent and sent[-1] == "." else sent for sent in sents]

        def create_samples_from_articles(articles):
            cur_res = []
            # if len(articles['steps'])>40: return []

            for article in articles:
                for i in range(1, len(article['steps']) - 1):
                    if i > 20 or len(article['steps']) - i > 20:
                        continue
                    cur_res.append({"src_text": article["title"] + self.SEP_TOKEN + self.SEP_TOKEN.join(clean_sents(article['steps'][:i])),
                                    "tgt_text": " ".join(article['steps'][i:])})

                    if args.add_finished:
                        cur_res[-1]["tgt_text"] += " Finished"

            return cur_res

        if args.rand_augment_train and in_train:
            print("rand_augment_train")
            # self.augmented_instances=[]

            source = load_file(args.rand_augment_train_file)
            source = np.random.choice(source, size=int(int(len(grounded) / 5) * args.rand_augment_train_ratio), replace=False)
            print("len(source)", len(source))

            created_samples = create_samples_from_articles(source)

            for k, item in enumerate(tqdm(created_samples)):
                src_text, tgt_text = item['src_text'], item['tgt_text']

                # tmp_tok_lens1 = len(self.tokenizer(self.SEP_TOKEN.join(cur_steps), padding=True, max_length=self.max_seq_len,
                #                                    truncation=True)['input_ids'])
                #
                # tmp_tok_lens2 = len(self.tokenizer(", ".join(tgt_steps), padding=True, max_length=self.max_seq_len,
                #                                    truncation=True)['input_ids'])

                model_inputs = self.tokenizer(src_text, padding=True, max_length=self.max_seq_len, truncation=True)

                with self.tokenizer.as_target_tokenizer():
                    # tgt_text = tgt_text.lower()
                    tgt = self.tokenizer(tgt_text, padding=True, max_length=self.max_seq_len,
                                         truncation=True)
                model_inputs["labels"] = tgt['input_ids']
                for key in ['input_ids', 'attention_mask']:
                    model_inputs["decoder_" + key] = tgt[key]
                model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

                # # don't do over long inputs
                exceed_max_len = False
                if max(len(tgt['input_ids']), len(model_inputs['input_ids'])) >= self.max_seq_len - 2:
                    exceed_max_len = True
                if exceed_max_len:
                    skipped += 1
                    print("\n skipped", skipped)
                    continue

                self.instances.append({
                    'tokenizer': tokenizer,
                    'global_doc_id': -1,
                    'doc_id': -1,
                    'path_id': -1,
                    'doc_type': "None",
                    'src_text': src_text.split("."),  # cur_steps,
                    'input_text': src_text,
                    'input_text_from_ids': tokenizer.batch_decode([model_inputs['input_ids']], skip_special_tokens=False)[0],
                    'tgt_text': tgt_text,
                    "sample_id": i + k,
                    'src_step_ids': [],
                    'tgt_step_ids': [],
                    "categories": "None",
                    "synthetic": 1,
                })
                self.instances[-1].update(model_inputs)

        """encode different parts into functions"""
        print("\nskipped", skipped)
        # save data
        if not generation_mode and args.cache_filename:
            # dump_file(records, os.path.join(args.subset_dir, "records_1.json"))

            dump_file(self.instances, save_file)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def convert_text_to_model_input(self, args, model_inputs, next_step):
        """

        :param next_step: str
        :return: dict of model input
        """

        # model_inputs = self.tokenizer(cur_history, max_length=self.max_seq_len, padding=True,
        #                               truncation=True)
        # input_ids, new_map, original_pos_vec, token_type_ids, attention_mask=sent_to_token_ids(cur_history, self.max_seq_len, self.tokenizer, shift_right=False)
        # model_inputs={}
        # model_inputs['input_ids']=input_ids
        # model_inputs['token_type_ids']=token_type_ids
        # model_inputs['attention_mask']=attention_mask

        def get_event_position_ids(lis, change_signals):
            cur_index = 0
            res = []
            for m, item in enumerate(lis):
                if m > 0 and (item in change_signals or lis[m - 1] in change_signals):
                    cur_index += 1
                res.append(cur_index)
            return res

        model_inputs["event_position_ids"] = get_event_position_ids(model_inputs["input_ids"], {self.SEP, self.CLS})

        if args.task_mode in ["gen", "ret"]:
            with self.tokenizer.as_target_tokenizer():
                tgt = self.tokenizer(next_step, padding=True,  # max_length=self.max_seq_len,
                                     truncation=True)

            # print("tgt", self.tokenizer.tokenize(next_step))
            # assert tgt['input_ids'][0]==self.tokenizer.cls_token_id, breakpoint()

            model_inputs["labels"] = tgt['input_ids']  # [1:]

            # model_inputs["subevent_pos"] = find_all_pos(model_inputs["input_ids"], self.SEP)
            # print("evt pos", model_inputs["event_position_ids"])
            assert len(model_inputs["event_position_ids"]), breakpoint()
            for key in ['input_ids', 'attention_mask']:
                model_inputs["decoder_" + key] = tgt[key]
            # if "bart" not in self.model_type:
            model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

        return model_inputs

    def convert_to_dict(self):
        needed_keys = ['input_ids', 'token_type_ids', 'attention_mask',
                       'decoder_input_ids', 'decoder_token_type_ids', 'decoder_attention_mask']
        ls = self.instances
        keys = needed_keys  # list(ls[0].keys()) #["model_inputs"] ["model_inputs"]
        res = {}
        for key in keys:
            res[key] = [i[key] for i in ls]

        # print("\nres.keys()", res.keys())
        return res

    def process_sample(self):
        pass

    @classmethod
    def get_category_to_steps(cls, files):
        category_to_steps = defaultdict(list)
        for f in files:
            data = load_file(f)
            for sample in data:
                for cat in sample['categories']:
                    category_to_steps[cat].append(sample['tgt_text'])

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance


###

def convert_text_to_model_input(model_inputs, next_step, tokenizer, SEP, CLS):
    """

    :param next_step: str
    :return: dict of model input
    """
    with tokenizer.as_target_tokenizer():
        tgt = tokenizer(next_step, padding=True,  # max_length=self.max_seq_len,
                        truncation=True)

    # print("tgt", self.tokenizer.tokenize(next_step))
    # assert tgt['input_ids'][0]==self.tokenizer.cls_token_id, breakpoint()

    model_inputs["labels"] = tgt['input_ids']  # [1:]

    def get_event_position_ids(lis, change_signals):
        cur_index = 0
        res = []
        for m, item in enumerate(lis):
            if m > 0 and (item in change_signals or lis[m - 1] in change_signals):
                cur_index += 1
            res.append(cur_index)
        return res

    model_inputs["event_position_ids"] = get_event_position_ids(model_inputs["input_ids"], {SEP, CLS})
    # model_inputs["subevent_pos"] = find_all_pos(model_inputs["input_ids"], self.SEP)
    # print("evt pos", model_inputs["event_position_ids"])
    assert len(model_inputs["event_position_ids"]), breakpoint()
    for key in ['input_ids', 'attention_mask']:
        model_inputs["decoder_" + key] = tgt[key]
    # if "bart" not in self.model_type:
    model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

    return model_inputs


def process_each_sample(cur_data):
    cur_sample_id, path, adj_data_sample, max_node_num, tokenizer, max_seq_len, SEP, CLS = cur_data

    # print("\n\n\nsample",i)
    global_doc_id = path["global_doc_id"]
    doc_id = path["doc_id"]
    path_id = path["path_id"]
    doc_type = path["doc_type"]
    doc_list = path["doc_list"]
    cur_steps = path["cur_steps"]
    step_types = path["step_types"]
    categories = path["categories"]
    first_qc = path["first_qc"]
    last_qc = path["last_qc"]
    useful_qc = set(first_qc) | set(last_qc)

    # concept_id_to_pos_list =  # list of dic, each dic for each step
    concept_id_to_pos_list = [{int(ccptid): dic[ccptid] for ccptid in dic} for dic in path["concept_id_to_pos_list"]]
    history_length = len(concept_id_to_pos_list)
    assert len(step_types) == history_length

    for j, dic in enumerate(concept_id_to_pos_list):
        for ccid in dic:
            dic[ccid] = set([tuple(pair) for pair in dic[ccid]])

    # print("doc_list", doc_list)
    # print("cur_steps", cur_steps)
    # print("concept_id_to_pos_list",concept_id_to_pos_list)
    # print("concept_id_to_pos_list name",[{cur_id2concept[ccptid]:dic[ccptid] for ccptid in dic} for dic in concept_id_to_pos_list])

    tgt_text = path["tgt_text"]
    # print("tgt_text", tgt_text)

    """========WM Input========="""
    """========Process Graph========="""

    """Do Sanity Check assert"""

    adj, concepts, _, _, connected_pairs, doc_id, global_doc_id, qc_ids, ac_ids = adj_data_sample
    # print("concepts", type(concepts))
    # print("concept node ids",concepts)
    # print("concept node names",[cur_id2concept[each_id] for each_id in concepts])
    conceptid2nodeid = {conceptid: idx for idx, conceptid in enumerate(concepts)} if concepts is not None else {}
    # print("conceptid2nodeid",conceptid2nodeid)
    # print("conceptid2nodeid names",{cur_id2concept[each_id]: conceptid2nodeid[each_id] for each_id in conceptid2nodeid})

    # connected_pairs=[[conceptid2nodeid[ccptid1], conceptid2nodeid[ccptid2]] for (ccptid1, ccptid2) in connected_pairs]
    if concepts is not None and concepts.size > 0:
        # print("concepts", concepts)
        ij = torch.tensor(adj.row, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
        tgt_axis = torch.tensor(adj.col,
                                dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
        n_node = adj.shape[1]
        # print("adj.shape[0] // n_node=", adj.shape[0] // n_node)
        half_n_rel = adj.shape[0] // n_node
        rel_axis, src_axis = ij // n_node, ij % n_node

        # print("half_n_rel", half_n_rel)
        # print("adj", adj)
        # print("rel_axis", rel_axis.shape)
        # print("src_axis", src_axis.shape)

        # real_n_nodes = len(concepts)
        mask = (src_axis < max_node_num) & (tgt_axis < max_node_num)
        src_axis, tgt_axis, rel_axis = src_axis[mask], tgt_axis[mask], rel_axis[mask]
        rel_axis, src_axis, tgt_axis = torch.cat((rel_axis, rel_axis + half_n_rel), 0), \
                                       torch.cat((src_axis, tgt_axis), 0), \
                                       torch.cat((tgt_axis, src_axis), 0)  # add inverse relations
        edge_index, edge_attr = torch.stack([src_axis, tgt_axis], dim=0), rel_axis

        concepts = concepts[:max_node_num]
        node_attr = torch.tensor(concepts, dtype=torch.long).unsqueeze(1)
        # print(node_attr.shape)

    else:
        print("concepts is none")
        # print("concepts", concepts)
        # edge_index, edge_attr=torch.tensor([[0],[0]],dtype=torch.long), torch.tensor([0],dtype=torch.long)
        # node_attr = torch.tensor([[]], dtype=torch.long)

        edge_index, edge_attr = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        node_attr = torch.tensor([], dtype=torch.long)
    # print("edge_index", edge_index)
    # print("edge_attr", edge_attr)
    # print("node_attr", node_attr)
    # concept_set=set(concepts)
    g_data = Data(x=node_attr,
                  edge_index=edge_index.long(),  # torch.tensor(edge_index, dtype=torch.long)
                  edge_attr=edge_attr.long())  # torch.tensor(, dtype=torch.long))  # .unsqueeze(1)
    assert g_data is not None, embed()

    """========Get pre_tokens (spacy), get mapping pre-token positions to concept ids, also have cls sep etc.========="""
    start_pos = 1
    token_pos_to_concept_ids = {}
    pre_tokens = [tokenizer.cls_token]
    for j, dic in enumerate(concept_id_to_pos_list):
        if j == 0 or j == len(concept_id_to_pos_list) - 1:
            for concept_id, pos_list in dic.items():
                for k, pos_range in enumerate(pos_list):
                    for pos in range(*pos_range):
                        if pos + start_pos not in token_pos_to_concept_ids:
                            token_pos_to_concept_ids[pos + start_pos] = set()
                        token_pos_to_concept_ids[pos + start_pos].update([concept_id])
        start_pos += 1 + len(doc_list[j])
        pre_tokens += doc_list[j]
        if j < len(concept_id_to_pos_list) - 1:
            if step_types[j] == "goal":
                pre_tokens += tokenizer.tokenize("[GOAL]")  # do not add SEP at end yet
                assert pre_tokens[-1] == "[GOAL]"

            elif step_types[j] == "subgoal":
                pre_tokens += tokenizer.tokenize("[SUBGOAL]")  # do not add SEP at end yet
                assert pre_tokens[-1] == "[SUBGOAL]"
            elif step_types[j] == "event":
                pre_tokens += [tokenizer.sep_token]  # do not add SEP at end yet
            else:
                print("unknown steptype appeared")
                embed()
    # print("token_pos_to_concept_ids", token_pos_to_concept_ids)
    # print("pre_tokens", pre_tokens)

    """========Get mapping token positions to node ids========="""

    # pre_tokenized_src_text needs to [ sep ]
    input_ids, new_map, _, token_type_ids, attention_mask = sent_to_token_ids(pre_tokens, max_seq_len,
                                                                              tokenizer,
                                                                              add_sep_at_end=True,
                                                                              has_cls_at_start=True)
    # print('sent_to_token_ids')
    # print("input_ids", input_ids)
    # print("new_map", new_map)

    #### convert pos_to_conceptids to index_to_conceptids
    ####[[2,-1,-1], [3,0,-1]] N_Nodes* max_num_matching_concepts
    #### in model forward, just +2 after concat

    concept_id_to_token_pos = {}
    token2nodepos = [set() for _ in range(len(input_ids))]
    for pre_token_pos, concept_ids in token_pos_to_concept_ids.items():
        for k in range(new_map[pre_token_pos], new_map[pre_token_pos + 1]):
            if k >= max_seq_len - 1:  # ignore SEP at the end
                break

            for concept_id in concept_ids:
                if not concept_id in concept_id_to_token_pos:
                    concept_id_to_token_pos[concept_id] = set()
                # print('k', k)
                concept_id_to_token_pos[concept_id].add(k)
            try:
                token2nodepos[k].update([conceptid2nodeid[int(concept_id)] for concept_id in concept_ids])
            except:
                print("token2nodepos update issue")
                embed()
    for j, item in enumerate(token2nodepos):
        token2nodepos[j] = list(item)
        if not token2nodepos[j]:
            token2nodepos[j].append(-1)
            # item.update([-1])  # default to random embedding

    # print("concept_id_to_token_pos", concept_id_to_token_pos)
    # print("token2nodepos", token2nodepos)
    # padding, -2 is 0 in embedding matrix, -1 is 1, > 1 are node embeddings
    # token2nodepos = pad_to_batch([sorted(lis) for lis in token2nodepos],
    #                              max_seq_len=args.max_concepts_num_for_each_token, pad_token_id=-2,
    #                              max_batch_seq_len=None)

    """========FSTM Input========="""

    # fstm_nodes = set()
    # for (ccptid1, ccptid2) in connected_pairs:
    #     fstm_nodes |= set(concept_id_to_token_pos[ccptid1]) | set(concept_id_to_token_pos[ccptid2])
    # fstm_nodes=sorted(fstm_nodes)
    fstm_nodes = list(range(len(input_ids)))

    fstm_token_pos_to_node_id = {token_pos: i for i, token_pos in enumerate(fstm_nodes)}
    edge_index = set()

    # tokens connecting to the same concept are fully connected
    for ccptid, token_pos in concept_id_to_token_pos.items():
        for pos1 in token_pos:
            for pos2 in token_pos:
                edge_index.add((pos1, pos2))
                edge_index.add((pos2, pos1))

    # self edge
    for ccptid in fstm_nodes:
        edge_index.add((ccptid, ccptid))
    # tokens one-hop away are connected
    for (ccptid1, ccptid2) in connected_pairs:
        if ccptid1 not in concept_id_to_token_pos or ccptid2 not in concept_id_to_token_pos: continue

        for token_pos1 in concept_id_to_token_pos[ccptid1]:
            for token_pos2 in concept_id_to_token_pos[ccptid2]:
                edge_index.add((fstm_token_pos_to_node_id[token_pos1], fstm_token_pos_to_node_id[token_pos2]))
                edge_index.add((fstm_token_pos_to_node_id[token_pos2], fstm_token_pos_to_node_id[token_pos1]))
        # except:
        #     print("fstm_nodes issue")
        #     embed()

    edge_index = np.array(sorted(list(edge_index))).T
    node_attr = torch.tensor(fstm_nodes, dtype=torch.long)  # .unsqueeze(1)
    g_data2 = Data(x=node_attr,
                   edge_index=torch.tensor(edge_index, dtype=torch.long))  # .unsqueeze(1)

    # print("node_attr2", node_attr)
    # print("edge_index2", edge_index)
    # print("edge_attr2", edge_attr)
    """========Prepare Model Input========="""
    model_inputs = {}
    model_inputs['input_ids'] = input_ids
    model_inputs['token_type_ids'] = token_type_ids
    model_inputs['attention_mask'] = attention_mask
    model_inputs = convert_text_to_model_input(model_inputs, tgt_text, tokenizer, SEP, CLS)
    model_inputs["g_data"] = g_data
    model_inputs["g_data2"] = g_data2
    model_inputs["token2nodepos"] = token2nodepos

    # print("model_inputs", model_inputs)
    assert len(model_inputs["token2nodepos"]) == len(model_inputs["input_ids"]), breakpoint()

    cur_res = {
        'tokenizer': tokenizer,
        'global_doc_id': global_doc_id,
        'doc_id': doc_id,
        'path_id': path_id,
        'doc_type': doc_type,
        'src_text': cur_steps,
        'tgt_text': tgt_text,
        "sample_id": cur_sample_id,
        # "history_length":history_length
        # "concept_set": concept_set
    }
    cur_res.update(model_inputs)
    return cur_res


@dataclass
class CustomCollatorCLF(DataCollatorWithPadding):
    collect_input: Optional[Any] = False
    collect_fields: Optional[List] = None

    def __call__(self, features):

        use_nbr_extra = "cbr" in features[0]['components']
        do_cat_nbs = "cat_nbs" in features[0]['components']
        # print("use_nbr_extra",use_nbr_extra)
        # print("do_cat_nbs",do_cat_nbs)

        if use_nbr_extra and not do_cat_nbs:
            # print("jas nbs_input_ids")

            encoder_features = []
            encoder_features_aggregate = []
            true_sample_indices = []

            for feat in features:
                true_sample_indices.append(len(encoder_features_aggregate))
                encoder_features_aggregate.append({'input_ids': feat['input_ids'],
                                                   'attention_mask': feat['attention_mask'],
                                                   })

                encoder_features_aggregate.extend([{'input_ids': feat['nbs_input_ids'][j],
                                                    'attention_mask': feat['nbs_attention_mask'][j],
                                                    } for j in range(len(feat['nbs_input_ids']))])
        elif use_nbr_extra and do_cat_nbs:
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
        if use_nbr_extra and not do_cat_nbs:
            print('use_nbr_extra and not do_cat_nbs:')

            aggregated_features = self.tokenizer.pad(
                encoder_features_aggregate,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            input_features.update({
                'aggregate_ipids': aggregated_features['input_ids'],
                'aggregate_atmsk': aggregated_features['attention_mask'],
            })
            # print("input_features bf", input_features)
            input_features['input_ids'] = input_features['aggregate_ipids'].index_select(0, torch.tensor(
                true_sample_indices, dtype=torch.long))
            input_features['attention_mask'] = input_features['aggregate_atmsk'].index_select(0, torch.tensor(
                true_sample_indices, dtype=torch.long))
            # print("input_features af", input_features)

        else:
            input_features.update(self.tokenizer.pad(
                encoder_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            ))
            # print("input_features for normal", input_features)

        input_features['labels'] = [feature["clf_label"] for feature in features]
        return input_features


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

        category_encoder_features = [{'input_ids': feat['category_input_ids'],
                                      'attention_mask': feat['category_attention_mask'],
                                      } for feat in features]
        input_features.update(modify_dict_keys(self.tokenizer.pad(
            category_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ), prefix="category_"))

        title_encoder_features = [{'input_ids': feat['title_input_ids'],
                                   'attention_mask': feat['title_attention_mask'],
                                   } for feat in features]
        input_features.update(modify_dict_keys(self.tokenizer.pad(
            title_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ), prefix="title_"))

        title2_encoder_features = [{'input_ids': feat['title2_input_ids'],
                                    'attention_mask': feat['title2_attention_mask'],
                                    } for feat in features]
        input_features.update(modify_dict_keys(self.tokenizer.pad(
            title2_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ), prefix="title2_"))

        input_features.update({
            'sample_ids': [feat['sample_id'] for feat in features]
        })

        return input_features
