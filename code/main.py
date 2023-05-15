from options import *
from train import train, train_clf, train_clf_trainer
from train_utils import *
from utils.utils import check_error2, get_best_ckpt
from evaluate import evaluate
from data import WikiHowDatasetEST, PrimitiveGenerationDataset
import wandb
import os
import shutil
import gc


def main_func():
    # global args
    """main"""
    """=========INIT========="""
    # wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))
    args = read_args()
    gc.collect()
    # print("args.nb_threshold", args.nb_threshold)

    if not args.debug and not args.task_mode == "ret":
        wandb.init()
        print("WANDB")

    """=========Set Tokenizer========="""
    tokenizer = get_tokenizer(args.plm, slow_connection=args.slow_connection)
    args.tokenizer= tokenizer

    if args.use_special_tag == 1: tokenizer.add_special_tokens({'additional_special_tokens': ['[GOAL]', '[SUBGOAL]', '[STEP]', '<ROOT>']})  # '[GOAL]','[SUBGOAL]'

    if args.max_seq_len == -1:
        args.max_seq_len = tokenizer.model_max_length if tokenizer.model_max_length <= 100000 else 512  # TODO
    print("args.max_seq_len", args.max_seq_len)

    """=========Set Generation Parameter========="""
    if args.script_completion or args.task_mode == "prt":
        # args.generated_max_length = args.max_seq_len-1
        args.generated_max_length = args.max_seq_len - 1

    """=========General Setup========="""
    # args, model, optimizer = setup_common(args, tokenizer)
    # print("setup done")
    # # print("args", args)
    args.abbr2category = abbr2category = {"food": "Food and Entertaining",
                                          "cv": "Cars & Other Vehicles", "fb":"Finance and Business"}
    if not args.eval:
        # full_seqs_data = load_file(args.full_seqs_file) if "cbr" in args.components and args.task_mode != "prt" else None
        # if args.debug:
        #     dev_data = WikiHowDataset(args, args.dev_file, args.dev_adj, tokenizer, full_seqs_data=full_seqs_data)
        #     breakpoint()

        """SPecial"""
        if False:  # args.pretrain_concept_generator
            train_data = PrimitiveGenerationDataset(args, args.train_file, tokenizer, in_train=True)
            dev_data = PrimitiveGenerationDataset(args, args.dev_file, tokenizer)
            test_data = PrimitiveGenerationDataset(args, args.test_file, tokenizer)
        else:

            if not args.pretrain_concept_generator:
                train_data = WikiHowDatasetEST(args, args.train_file, tokenizer, tcg_file=args.train_tcg,
                                               step_db_file=args.train_step_db, in_train=True)
                # print("train_data", len(train_data.instances))
                dev_data = WikiHowDatasetEST(args, args.dev_file, tokenizer, tcg_file=args.dev_tcg,
                                             step_db_file=args.dev_step_db)
                test_data = WikiHowDatasetEST(args, args.test_file, tokenizer, tcg_file=args.test_tcg,
                                              step_db_file=args.test_step_db)
            else:
                train_data = dev_data = PrimitiveGenerationDataset(args, args.dev_file, tokenizer)
                test_data = PrimitiveGenerationDataset(args, args.test_file, tokenizer)
                if not args.eval_only:
                    train_data = PrimitiveGenerationDataset(args, args.train_file, tokenizer, in_train=True)
                    # dev_data = PrimitiveGenerationDataset(args, args.dev_file, tokenizer)

        if args.top_few != -1:
            train_data.instances = train_data.instances[: args.top_few]
            dev_data.instances = dev_data.instances[: args.top_few]
            test_data.instances = test_data.instances[: args.top_few]

        # for dt in [train_data, dev_data, test_data]:
        #     dt.instances=[sample for sample in dt.instances if len(sample["ent_list"])]
        # dt.instances=dt.instances[:int(len(dt.instances)//2)]

        # embed()

        if args.only_cache_data:  # no need to run program
            return

        if False:  # "fssss" in args.data_mode
            print("in fs")
            print("len(train_data.instances) bef", len(train_data.instances))

            fs_indices = set(load_file(args.subset_dir + f"{args.data_mode}.json"))
            train_data.instances = [sample for sample in train_data.instances if sample["global_doc_id"] in fs_indices]
            print("len(train_data.instances) aft", len(train_data.instances))

        elif "longseq" in args.data_mode:
            dev_data = [sample for sample in dev_data if len(sample["cur_steps"]) > 10]
            test_data = [sample for sample in test_data if len(sample["cur_steps"]) > 10]

        elif "crossdomain" in args.data_mode:
            train_indices = set(load_file(args.subset_dir + f"data_subset1_cleaned_filtered_{args.data_mode}_indices_train.json"))
            dev_indices = set(load_file(args.subset_dir + f"data_subset1_cleaned_filtered_{args.data_mode}_indices_dev.json"))
            test_indices = set(load_file(args.subset_dir + f"data_subset1_cleaned_filtered_{args.data_mode}_indices_test.json"))
            tmp_train_data, tmp_dev_data, tmp_test_data = [], [], []
            for ds in [train_data, dev_data, test_data]:
                for sample in ds:
                    if sample['global_doc_id'] in train_indices:
                        tmp_train_data.append(sample)
                    elif sample['global_doc_id'] in dev_indices:
                        tmp_dev_data.append(sample)
                    elif sample['global_doc_id'] in test_indices:
                        tmp_test_data.append(sample)
        elif args.data_mode in abbr2category:
            all_data = [train_data, dev_data, test_data]
            print(f"len(train_data.instances) bef", len(train_data.instances))
            for j, ds in enumerate(all_data):
                ds.instances = [sample for sample in ds.instances if abbr2category[args.data_mode] in sample["categories"]]
            print(f"len(train_data.instances) aft", len(train_data.instances))

            # train_data = [sample for sample in train_data if abbr2category[args.data_mode] in sample["categories"]]
            # dev_data = [sample for sample in dev_data if abbr2category[args.data_mode] in sample["categories"]]
            # test_data = [sample for sample in test_data if abbr2category[args.data_mode] in sample["categories"]]

        # if args.fs_mode:
        #     ssize = int(args.fs_mode[2:])
        #     train_data.instances = train_data.instances[:ssize]
        #     # dev_data.instances = dev_data.instances#[:ssize]
        #     # test_data.instances = test_data.instances#[:ssize]
        """length control"""

        if len(args.history_length):
            print("len(args.history_length)", len(args.history_length))
            completion_length_start, completion_length_end = args.completion_length.split(",") if args.completion_length else (0, 0)
            history_length_start, history_length_end = args.history_length.split(",") if args.history_length else (0, 0)
            all_data = [train_data, dev_data, test_data]
            print(f"len(train_data.instances) bef", len(train_data.instances))
            for j, ds in enumerate(all_data):
                tmp = []
                for sample in ds:
                    # valid_tgt_length= args.completion_length != "" and (len(sample) < int(completion_length_start) or len(tgt_step_ids) >= int(completion_length_end)):
                    # if args.history_length != "" and (len(src_step_ids) < int(history_length_start) or len(src_step_ids) >= int(history_length_end)):
                    #     continue
                    if not args.history_length or (int(history_length_start) <= len(sample['src_text']) < int(history_length_end)):
                        tmp.append(sample)
                ds.instances = tmp
            print(f"len(train_data.instances) aft", len(train_data.instances))

        args, model, optimizer = setup_common(args, tokenizer)
        print("setup done")
        # if args.check_data:
        #     for ds in [train_data, dev_data, test_data]:
        #         check_error2(ds.instances)
        if args.debug:
            # print("args.debug", args.debug)
            train_data.instances, dev_data.instances, test_data.instances = train_data.instances[:8], dev_data.instances[:4], test_data.instances[:4]
        if args.debug or args.detect_anomaly:
            print("set detect_anomaly")
            torch.autograd.set_detect_anomaly(True)
        if args.task_mode in ["gen", "prt"]:
            train(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))
        elif args.task_mode == "clf":
            # train_clf(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))
            train_clf_trainer(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))



    else:

        test_data = WikiHowDataset(args, args.test_file, args.test_adj, tokenizer)
        # concept_index_mapping=get_concept_index_mapping([test_data])
        concept_index_mapping = None

        args, model, optimizer = setup_common(args, tokenizer, concept_index_mapping)
        print("setup done")

        print("Eval Mode")
        # best_ckpt=get_best_ckpt()
        # model = model.from_pretrained(best_ckpt).to(args.device)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        evaluate(args, model, test_data, tokenizer, no_scoring=True)

    # remove all train states due to space

    # with os.scandir(args.output_dir) as entries:
    #     for entry in entries:
    #         if entry.is_dir() and not entry.is_symlink():
    #             shutil.rmtree(entry.path)
    #         else:
    #             os.remove(entry.path)

    if not args.debug:
        print("wandb.config2", dict(wandb.config))
    # arg_dict=vars(args)
    # for key in wandb.config.keys():
    #     if key not in args.tunable_params:
    #         wandb.config._items.pop(key, None)

    # embed() #wandb.config._items.pop("_n_gpu",None)


if __name__ == '__main__':
    # with launch_ipdb_on_exception():
    #     main_func()
    main_func()