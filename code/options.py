import argparse
# import configparser
# from utils.utils import load_file
# from IPython import embed
# from pprint import pprint as pp
import os
import torch
import random
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu_id", default="", type=str, help="gpu_id", )
# args, _ = parser.parse_known_args()
# if len(args.gpu_id):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#     print("gpuids", os.environ["CUDA_VISIBLE_DEVICES"])

# from multiprocessing import cpu_count
# from multiprocessing import Pool
EMB_PATHS = {
    'transe': 'data/transe/glove.transe.sgd.ent.npy',
    'lm': 'data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': 'data/transe/concept.nb.npy',
    'tzw': 'data/cpnet/tzw.ent.npy',
}
PLM_DICT = {
    "bert-mini": "prajjwal1/bert-mini",
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-base-cased": "bert-base-cased",
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-cased": "bert-large-cased",
    "bert-large-uncased": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "sap": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
    "sci": "allenai/scibert_scivocab_uncased",
    "gpt2": "gpt2",
    "bart-tiny": "sshleifer/bart-tiny-random",
    "bart-base": "facebook/bart-base",
    "bart-large": "facebook/bart-large",
    "mt5-tiny": "stas/mt5-tiny-random",
    "t5-tiny": "patrickvonplaten/t5-tiny-random",
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
}
PLM_DIM_DICT = {
    "bert-mini": 256,
    "bert-tiny": 128,
    "bert-small": 512,
    "bert-medium": 512,
    "bert-base-cased": 768,
    "bert-base-uncased": 768,
    "bert-large-cased": 1024,
    "bert-large-uncased": 1024,
    "roberta-base": 1024,
    "roberta-large": 1024,
    "sap": 768,
    "sci": 768,
    "bart-tiny": 24,
    "bart-base": 768,
    "bart-large": 1024,
    "t5-tiny": 64,
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024
}
PLM_LEN_DICT = {
    "bert-mini": 512,
    "bert-tiny": 512,
    "bert-small": 512,
    "bert-medium": 512,
    "bert-base-cased": 512,
    "bert-base-uncased": 512,
    "roberta-base": 1024,
    "roberta-large": 1024,
    "sap": 512,
    "sci": 512,
    "bart-tiny": 1024,
    "bart-base": 1024,
    "bart-large": 1024,
    "t5-tiny": 1024,
    "t5-base": 1024,
    "t5-large": 1024
}




def add_generation_arguments(parser):
    parser.add_argument("--length_penalty", default= 1.2, type=float)
    parser.add_argument("--num_beams", default=5, type=int)
    # parser.add_argument("--do_sample", default=1, type=int)


    parser.add_argument("--generated_max_length", default=90, type=int)
    parser.add_argument("--generated_min_length", default=2, type=int)
    args, _ = parser.parse_known_args()


def add_experiment_arguments(parser):
    parser.add_argument("--output_dir", default="", type=str)
    # parser.add_argument("--experiment", default="exp", type=str)
    parser.add_argument("--experiment_path", default="../experiment/", type=str)
    parser.add_argument("--exp", default="exp", type=str)
    parser.add_argument("--analysis_dir", default="analysis/", type=str)
    parser.add_argument("--use_gpu", default=1, type=int, help="Using gpu or cpu", )
    args, _ = parser.parse_known_args()
    # parser.add_argument("--exp_id", default="0", type=str)
    parser.set_defaults(exp_id=os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else "0")
    num_devices=1
    if args.use_gpu:
        num_devices=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if 'CUDA_VISIBLE_DEVICES' in os.environ else torch.cuda.device_count()

    if not os.path.exists(args.analysis_dir):
        os.mkdir(args.analysis_dir)
    parser.set_defaults(my_num_devices=num_devices)
    print("numdevice", num_devices)

    parser.add_argument("--static_graph", default=0, type=int)
    parser.add_argument("--use_event_tag", default=0, type=int)
    parser.add_argument("--task_mode", default="gen",  type=str) # clf
    parser.add_argument("--script_completion", default=1,  type=int) # clf
    # parser.add_argument("--script_completion", default=0,  type=int)
    parser.add_argument("--intra_event_encoding", default=1,  type=int) # clf
    parser.add_argument("--no_dl_score", default=1,  type=int) # clf
    parser.add_argument("--visualize_scatterplot", action="store_true") # clf
    parser.add_argument("--save_output", action="store_true") # clf
    parser.add_argument("--update_input", action="store_true") # clf
    parser.add_argument('--scatterplot_metrics', default=["rouge", "bleu", "meteor"], nargs='+')
    parser.add_argument("--train_gold_concepts", default=0,  type=int) # clf
    parser.add_argument("--unordered_gold_concepts", default=0,  type=int) # clf
    parser.add_argument("--rand_augment_train", default=0,  type=int) # clf
    parser.add_argument("--rand_augment_train_ratio", default=0.3,  type=float) # clf
    parser.add_argument("--rank_mode", default="orig_rank_g",  type=str) # clf
    parser.add_argument("--ct", default="v2",  type=str) # clf
    parser.add_argument("--cl_exact_location", default=0,  type=int) # clf
    parser.add_argument("--gold_percent", default="",  type=str) # clf
    parser.add_argument("--cl_exp_type", default="",  type=str) # clf



    parser.add_argument("--exp_msg", default="",  type=str) # clf
    parser.add_argument("--label_mode", default="multiclass",  choices=["multiclass", "multilabel", "binary"],type=str) # multilabel or not # todo

    tunable_params=['model_name', 'plm', 'subset', 'components', 'use_special_tag','metric_for_best_model','greater_is_better',
                        'length_penalty', 'num_beams', 'generated_max_length', 'generated_min_length', 'has_concepts','patience',
                            'num_epochs', 'batch_size','true_batch_size', 'eval_batch_size', 'plm_lr', 'lr', 'scheduler',
                            'warmup_ratio', 'eval_steps', 'eval_strategy', 'label_smoothing_factor','num_evals_per_epoch',
                            'g_dim','pool_type',  'num_gnn_layers', 'num_gnn_layers2', 'gnn_type','gnn_type2',"optim",'use_cache','max_num_ents',
                        'init_range',  'decattn_layer_idx', 'freeze_ent_emb', 'max_node_num','gpu_id','use_gpu','exp_msg' , 'script_completion', "no_test",'ret_components','nb_threshold','data_mode','seed','use_token_tag','cat_text_embed',
                    'intra_event_encoding', 'temporal_encoding','propagate_factor_embeddings']

    parser.set_defaults(
        # rare_params=['data_dir', 'use_cache', 'subset', 'ent_emb', 'max_node_num', 'train_file', 'dev_file', 'test_file',
        #              'train_adj', 'dev_adj', 'test_adj', 'weight_decay', 'adam_epsilon', 'max_grad_norm', 'use_gpu', 'use_amp', 'grad_accumulation_steps',
        #              'optim', 'model_path', 'downstream_layers', 'dropoute', 'dropouti', 'gnn_type', 'gnn_type2', 'freeze_ent_emb', 'debug', 'plm_hidden_dim'],
        tunable_params=sorted(set(tunable_params))
    )

def add_data_arguments(parser):
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--lower_case_tgt", default=0, type=int)
    parser.add_argument("--has_tgt_type", default=0, type=int)
    # parser.add_argument("--", default=0, type=int)

    args, _ = parser.parse_known_args()
    # global glob_seed
    # glob_seed = args.seed


    parser.add_argument("--num_workers", default=0, type=int)
    # parser.add_argument("--num_processes", default=cpu_count(), type=int)
    parser.add_argument("--out_dim", default=1, type=int)


    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--dataset", default="wikihow", type=str)
    # parser.add_argument("--train_file", default="", type=str)
    # parser.add_argument("--dev_file", default="", type=str)
    # parser.add_argument("--test_file", default="", type=str)

    parser.add_argument("--cache_filename", default="", type=str)
    parser.add_argument("--index_filename", default="", type=str)
    parser.add_argument("--step_db_filename", default="", type=str)

    parser.add_argument("--use_cache", default=0, type=int)
    parser.add_argument("--use_mat_cache", default=0, type=int)

    parser.add_argument("--check_data", default=0, type=int)
    parser.add_argument("--subset", default="subset9", type=str)
    parser.add_argument('--ent_emb', default=['tzw'], choices=['tzw', 'transe'], nargs='+', help='sources for entity embeddings')
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--max_concepts_num_for_each_token', default=2, type=int)
    parser.add_argument('--has_concepts', default=0, type=int)
    parser.add_argument('--use_special_tag', default=3, type=int)
    parser.add_argument('--num_nbs', type=int, default=2)
    parser.add_argument('--num_edge_types', default=12, type=int)
    parser.add_argument('--num_node_types', default=-1, type=int)
    parser.add_argument('--max_num_ents', default=10, type=int)
    parser.add_argument('--completion_length', default="", type=str)
    parser.add_argument('--hierachy', default=0, type=int)
    parser.add_argument('--num_tgt_steps',  default=-1, type=int)
    parser.add_argument('--factor_expander',  default=0, type=int)
    parser.add_argument('--augment_data',  default=0, type=int)
    parser.add_argument('--pretrain_concept_generator',  default=0, type=int)
    parser.add_argument('--use_generated_factors',  default=0, type=int)
    parser.add_argument("--custom_test_file", default="", type=str)
    parser.add_argument("--custom_dev_file", default="", type=str)
    parser.add_argument('--data_mode', default="full",  type=str)
    parser.add_argument('--add_finished',  default=1, type=int)
    parser.add_argument('--special_prompt',  default=1, type=int)
    parser.add_argument('--topk_prompt',  default=2, type=int)
    parser.add_argument('--prompt_at_back',  default=0, type=int)
    parser.add_argument('--eval_gold_concepts',  default=0, type=int)
    parser.add_argument('--silver_eval',  default=0, type=int)
    parser.add_argument('--use_pretrained_concepts',  default=0, type=int)
    parser.add_argument("--fs_mode", default="", type=str)
    parser.add_argument('--fs_split_id',  default=-1, type=int)
    parser.add_argument('--do_cl',  default=0, type=int)
    parser.add_argument('--use_cbr_text',  default=0, type=int)
    parser.add_argument('--nb_collect_mode',  default=0, type=int) #union
    parser.add_argument('--fill_empty_with_cg',  default=0, type=int) #union
    parser.add_argument('--fill_empty_neg_with_null_str',  default=1, type=int) #union
    parser.add_argument('--history_length', default="", type=str)
    parser.add_argument('--hist_length', default="", type=str)




    #choices=['fs0.1', 'fs0.3', 'fs0.5', 'fs0.5',"crossdomain"],


    args, _ = parser.parse_known_args()



    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb])
    # print("args.ent_emb", args.ent_emb)

    prefix_dict = {
        "full": "data_",
        "subset1": "data_subset1_cleaned_filtered_",
        "subset2": "data_subset2_cleaned_filtered_",
        "subset3": "data_",
        "subset4": "data_",
        "subset5": "data_",
        "subset6": "data_",
        "subset7": "data_",
        "subset8": "data_",
        "subset9": "cb_full_data_", #full_data_ #nnewdata_
        "subset10": "data_",
        "subset11": "data_",
        "subset12": "data_",
        "subset13": "data_",
        "subset14": "data_",
        "full_cgen": "data_",
        "gosc": "data_",
        "debug": "data_",
    }
    if args.silver_eval:
        prefix_dict["subset9"] = "newdata_"
    # if args.use_pretrained_concepts:
    #     prefix_dict["subset9"] = "new_data_"

    grounded_suffix,train_grounded_suffix="",''
    if args.task_mode == "gen":
        # train_grounded_suffix = grounded_suffix = ".grounded.nbs.json"
        train_grounded_suffix = grounded_suffix = ".grounded.json"
        if "goodnbsfs" in args.data_mode:
            train_grounded_suffix = f".grounded.{args.data_mode}.json"
            grounded_suffix = f".grounded.goodnbs.json"
        elif "fs" in args.data_mode:
            train_grounded_suffix = f".grounded.{args.data_mode}.json"
        elif "recipe" in args.data_mode:
            train_grounded_suffix=grounded_suffix = f".grounded.{args.data_mode}.json"
    elif args.task_mode == "ret":
        train_grounded_suffix = grounded_suffix = ".grounded.json"
    elif args.task_mode == "clf":
        train_grounded_suffix = grounded_suffix = ".grounded.clf.nbs.json"

    # elif args.task_mode == "prt":
    #     train_grounded_suffix = grounded_suffix = ".grounded.clf.nbs.json"

    # print("train_grounded_suffix", train_grounded_suffix)
    # print("grounded_suffix", grounded_suffix)
    # grounded_suffix = ".grounded.json" if args.task_mode!="clf" else ".grounded.clf.json"# .grounded.pkl
    graph_suffix = ".graph.adj.pk"

    subset_dir = args.data_dir + args.dataset + "/" + args.subset + "/"
    # train_file = subset_dir + prefix_dict[args.subset] + "train.json",
    # dev_file = subset_dir + prefix_dict[args.subset] + "dev.json",
    # test_file = subset_dir + prefix_dict[args.subset] + "test.json",
    train_step_db = dev_step_db = test_step_db = None
    train_tcg = dev_tcg = test_tcg = None
    if args.pretrain_concept_generator:#args.pretrain_concept_generator
        train_file=f'{subset_dir}full_non_eval_data_train.json' #full_non_eval_data_train data_train
        dev_file=f'{subset_dir}data_dev.json'
        test_file=f'{subset_dir}data_test.json'

    else:
        # if args.task_mode == "prt":
        #     train_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}train.grounded_pretrain_train.json'
        #     dev_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}train.grounded_pretrain_dev.json'
        #     # test_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}train{grounded_suffix}'
        #     test_file=dev_file
        # else:
        if args.fs_mode:
            # fs_n_samples,fs_split_id= args.fs_mode.split()
            fs_n_samples,fs_split_id= args.fs_mode, args.fs_split_id
            train_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}train_fs_{args.data_mode}_{fs_n_samples}_{fs_split_id}.json'
        else:
            train_file = f'{subset_dir}grounded/{prefix_dict[args.subset]}train{train_grounded_suffix}'

        # if args.do_cl:
        # cl_suffix=f"_{args.cl_version}" if args.cl_version else ""
        cl_suffix=f"_v2"

        if True: #args.gold_percent
            train_file = f'{subset_dir}grounded/{prefix_dict[args.subset]}train_v4_goldperct.json' #{cl_suffix} .grounded
            dev_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}dev_v4_goldperct.json'
            test_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}test_v4_goldperct.json'
        else :
            train_file = f'{subset_dir}grounded/{prefix_dict[args.subset]}train_v4.json' #{cl_suffix} .grounded
            dev_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}dev.json'
            test_file=f'{subset_dir}grounded/{prefix_dict[args.subset]}test.json'
        # from utils.utils import *
        # a=load_file(dev_file)
        # a=load_file(train_file)


        cgen_dir="data/wikihow/full_cgen/"
        train_tcg=f'{cgen_dir}data_train.json' #full_non_eval_data_train data_train
        dev_tcg=f'{cgen_dir}data_dev.json'
        test_tcg=f'{cgen_dir}data_test.json'

        step_db_filename=train_file
        train_step_db=f'{subset_dir}separated/{prefix_dict[args.subset]}train.json'
        dev_step_db=f'{subset_dir}separated/{prefix_dict[args.subset]}dev.json'
        test_step_db=f'{subset_dir}separated/{prefix_dict[args.subset]}test.json'



    parser.set_defaults(
        rand_augment_train_file=f'{subset_dir}non_tr_eval_data.json',
        subset_dir=subset_dir,
        train_file=train_file,
        dev_file=dev_file if not args.custom_dev_file else args.custom_dev_file,
        test_file=test_file if not args.custom_test_file else args.custom_test_file,
        train_tcg=train_tcg,
        dev_tcg=dev_tcg,
        test_tcg=test_tcg,
        train_step_db=train_step_db,
        dev_step_db=dev_step_db,
        test_step_db=test_step_db,
        save_dev_location=f"{subset_dir}dev_concepts.json",
        save_test_location=f"{subset_dir}test_concepts.json",
        # train_file_nbs=f'{subset_dir}grounded/{prefix_dict[args.subset]}train.grounded.json',
        # dev_file_nbs=f'{subset_dir}grounded/{prefix_dict[args.subset]}dev.grounded.json',
        # test_file_nbs=f'{subset_dir}grounded/{prefix_dict[args.subset]}test{grounded_suffix}',
        # full_seqs_file=f'union/grounded/full_seqs.json',
        full_seqs_file=f'{subset_dir}/full_seqs.json',
        train_adj=f'{subset_dir}graph/{prefix_dict[args.subset]}train{graph_suffix}',
        dev_adj=f'{subset_dir}graph/{prefix_dict[args.subset]}dev{graph_suffix}',
        test_adj=f'{subset_dir}graph/{prefix_dict[args.subset]}test{graph_suffix}'

    )


def add_training_arguments(parser):
    parser.add_argument("--num_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--true_batch_size", default=-1, type=int, help="True Batch size for training.") # todo
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training.")

    args, _ = parser.parse_known_args()
    # parser.set_defaults(true_batch_size=args.batch_size) #, total_batch_size=args.batch_size*args.num_devices

    parser.add_argument("--eval_batch_size", default=12, type=int, help="Batch size for training.")
    parser.add_argument("--plm_lr", default=3e-5, type=float, help="The initial learning rate for PLM.")
    parser.add_argument("--lr", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--metric_for_best_model", default='bleu', type=str, ) #todobertscore
    args, _ = parser.parse_known_args()
    if args.task_mode == "clf":
        metric_for_best_model="f1"
        greater_is_better=1
    elif args.task_mode == "gen":
        greater_is_better = 1
        if args.metric_for_best_model in ["loss"]:
            greater_is_better=1
    elif args.task_mode == "prt":
        metric_for_best_model="loss"
        greater_is_better=0
    else:
        assert False
    parser.set_defaults(greater_is_better=greater_is_better)



    parser.add_argument("--scheduler", default="linear",type=str, )  # constant_with_warmup  constant
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")

    parser.add_argument("--eval_steps", default=1000, type=int, help="Number of steps between each evaluation.")
    parser.add_argument("--num_evals", default=5, type=int, help="Number of eavls")
    parser.add_argument("--num_evals_per_epoch", default=2, type=float, help="Number of eavls")
    parser.add_argument("--eval_strategy", default="steps", type=str)  # epoch
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--burn_in', type=int, default=0)

    parser.add_argument("--label_smoothing_factor", default=0.0, type=float, )
    # parser.add_argument('-l', '--list', nFargs='+', help='<Required> Set flag', required=True)

    parser.add_argument("--gpu_id", default="", type=str, help="gpu_id", )
    args, _ = parser.parse_known_args()
    if len(args.gpu_id):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print("gpuids", os.environ["CUDA_VISIBLE_DEVICES"])


    parser.add_argument("--n_gpu", default=1, type=int, help="Number of gpu", )
    # parser.add_argument("--use_gpu", default=1, type=int, help="Using gpu or cpu", )
    parser.add_argument("--use_amp", default=0, type=int, help="Using mixed precision")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'])

    parser.add_argument("--prep_batch_size", default=40, type=int)
    parser.add_argument("--pred_with_gen", default=1, type=int)
    parser.add_argument("--do_eval", default=1, type=int)


def add_model_arguments(parser):
    parser.add_argument("--model_name", default="bsl_bart", type=str) #, choices=['bsl_bert', 'bsl_bart', 'bsl_t5', 'bsl_gpt2', 'see', 'retriever']
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--wandb_run_id", default="", type=str)
    parser.add_argument("--load_model_path", default="", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--temporal_encoding", default="lstm", type=str)
    parser.add_argument("--ablation", default=0, type=int) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--num_tag_types", default=2, type=int) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--use_token_tag", default=0, type=int) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--cat_text_embed", default=0, type=int) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--cl_type", default=1, type=int) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--cl_weight", default=0.3, type=float) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--cl_sample_types", default="12", type=str) # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--cl_empty_mode", default=1, type=int) # 0: replace with type 4  steps db  1: 02 as label 2: -100 as label
    parser.add_argument("--orig_cl_way", default=1, type=int) # 0: replace with type 4  steps db  1: 02 as label 2: -100 as label



    parser.add_argument("--plm", default="bart-base", type=str, metavar='N')  # default="base-uncased",
    args, _ = parser.parse_known_args()

    print("cl_sample_types", args.cl_sample_types)

    plm_class=args.plm.split("-")[0] #.split("/")[1]
    if "cased" in args.plm:
        plm_class+="_cased"
    elif "uncased" in args.plm:
        plm_class+="_uncased"
    parser.set_defaults(plm_class=plm_class) #



    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument('--downstream_layers', default=["nonenone","coherence_classifer"], nargs='+')#,"coherence_classifer"
    parser.add_argument('--no_decay_layers', default=['bias', 'LayerNorm.bias', 'LayerNorm.weight'], nargs='+') #layer_norm.bias layer_norm.weight

    module_choices = ["event_tag", "wm", "fstm", "cm", "ca", "none"]
    module_choices = ["match", "graph", "cbr", "cat_nbs", "none"]
    parser.add_argument('--components', type=str, default="")  # ca is cross attention b/w memory, cm is causal memory
    # parser.add_argument('--components', type=str, default=module_choices, choices=module_choices, nargs='+')  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--remove_modules', type=str, default=[], choices=module_choices, nargs='+')  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--ret_components', type=str, default="title")  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--nb_threshold', type=float, default=0.99986)  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--decattn_layer_idx', type=str, default='all')  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--propagate_factor_embeddings', type=int, default=0)  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--freeze_plm', default=0, type=int, help='freeze plm embedding layer')



    # parser.add_argument('--ret_components', default=["title", "cat", "seq"], choices=module_choices, nargs='+')  # ca is cross attention b/w memory, cm is causal memory


    # parser.add_argument('--concept_dim', type=int, default=256, help='Number of final hidden units for graph.')
    # parser.add_argument('--plm_hidden_dim', type=int, default=768, help='Number of hidden units for plm.')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Number of embedding units.')
    parser.add_argument('--g_dim', type=int, default=-1, help='Number of final hidden units for graph.')
    # parser.add_argument('--g_dim2', type=int, default=256, help='Number of final hidden units for graph.')

    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout")
    parser.add_argument('--dropoute', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    parser.add_argument("--activation", default="gelu", type=str)  # gelu
    parser.add_argument('--batch_norm', default=False, help="Please give a value for batch_norm")
    parser.add_argument('--pool_type', default="mean", type=str)  # for cm, 0 mean max, 1 max mean, 2 mean, 3 max

    parser.add_argument('--gnn_type', default="gine")  # "gat", "gine", "gin", , choices=["rgcn", "compgcn", "fast_rgcn"]
    parser.add_argument('--gnn_type2', default="gine") #, choices=["gat", "gin"]
    # parser.add_argument('--g_global_pooling', default=0, type=int)
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of final hidden units for graph.')
    parser.add_argument('--num_gnn_layers2', type=int, default=2, help='Number of final hidden units for graph.')

    parser.add_argument('--freeze_ent_emb', default=1, type=int, help='freeze entity embedding layer')
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')
    parser.add_argument('--n_attention_head', type=int, default=8)


def add_extra_arguments(parser):

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_r", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--only_cache_data", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--run_dev", action="store_true")

    parser.add_argument("--no_test", type=int, default=0)
    parser.add_argument("--top_few",  type=int, default=-1)
    parser.add_argument("--save_mat", action="store_true")

    # parser.add_argument("--debug", default=1, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--analyze", default=0, type=int)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--slow_connection", action="store_true")


def read_args():
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_training_arguments(parser)
    add_generation_arguments(parser)
    add_extra_arguments(parser)

    args = parser.parse_args()

    # print("args.ret_components bf", args.ret_components)
    args.ret_components=[item.strip() for item in args.ret_components.split(",") if item.strip()]
    # print("args.ret_components af", args.ret_components)

    # print("args.components bf", args.components)
    args.components=[item.strip() for item in args.components.split(",") if item.strip()]
    print("args.components af", args.components)
    #
    # tmp=args.data_mode.split()
    # if len(tmp)>1: args.data_mode, args.fs_mode=tmp
    # else: args.data_mode=tmp[0]

    "======DEBUG======"
    if args.debug:
        print("Debug Mode ON")
        # parser.set_defaults(plm="bert-tiny", batch_size=2, num_epochs=3, patience=3, eval_steps=5)
        # args.plm = "bart-tiny"
        # args.test_file = args.dev_file
        # args.train_file = args.dev_file
        # # args.test_step_db = args.dev_step_db
        # args.train_step_db =args.dev_step_db
        # # args.test_tcg = args.dev_tcg
        # args.train_tcg =args.dev_tcg

        args.batch_size = 2
        args.eval_batch_size = 2
        args.true_batch_size = args.batch_size
        # args.num_epochs = 1
        args.patience = 5
        args.eval_steps = 6


    if args.true_batch_size ==-1: args.true_batch_size = args.batch_size
    if args.cl_exp_type=="a":
        args.cl_empty_mode=0
        args.fill_empty_neg_with_null_str=0
        args.cl_sample_types="12"
    elif args.cl_exp_type=="b":
        args.cl_empty_mode=1
        args.fill_empty_neg_with_null_str=1
        args.cl_sample_types="23"


    "======PLM Model and Dim======"
    args.plm_hidden_dim = PLM_DIM_DICT[args.plm]
    # if args.g_dim == -1:
    #     args.g_dim = args.plm_hidden_dim
    args.plm = PLM_DICT[args.plm]
    print("PLM Model is", args.plm)


    # if "-tiny" in args.plm:
    #     args.plm_hidden_dim=128
    # elif "-mini" in args.plm:
    #     args.plm_hidden_dim=256
    # elif "-small" in args.plm or "-medium" in args.plm:
    #     args.plm_hidden_dim=512
    # elif "-large" in args.plm:
    #     args.plm_hidden_dim=1024
    # else:
    #     args.plm_hidden_dim=768

    # print("plm_hidden_dim  is", args.plm_hidden_dim)


    random.seed(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # pp(args)tunable_params

    return args
