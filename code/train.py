# from transformers import BertTokenizer
# from rouge_score import rouge
# import datasets
# from transformers import BertTokenizerFast
# import os
# os.environ['WANDB_DISABLED'] = "true"
import gc
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from datasets import load_metric
from torch.utils.data import DataLoader
from eval_final import Evaluate

from evaluate import *
# from torch.optim.lr_scheduler import _LRScheduler
import wandb
# from data import CustomCollator, CustomCollatorCLF  # , collate_wrapper
from data_collator import CustomCollator

# import wandb
# from transformers import get_linear_schedule_with_warmup
from evaluate import evaluate_clf, get_scores_multilabel_clf
# from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
# from datasets import Dataset, load_dataset
# from data import pad_to_batch
from train_utils import get_scheduler, get_tensor_float
from train_utils import seed_worker
from transformers import PreTrainedTokenizerBase
from transformers import SchedulerType
# from train_utils import seed_worker
# from utils import load_file, dump_file, visualize_plot
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainingArguments
from transformers.trainer import Trainer
from utils.utils import modify_dict_keys
from utils.data_utils import *
import time
from BARTScore.bart_score import BARTScorer
from constants import *

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        # loss_fct =  torch.nn.BCEWithLogitsLoss()
        loss_fct = torch.nn.MultiLabelSoftMarginLoss()
        # print("logits", get_tensor_info(logits))
        # print("self.model.num_labels", self.model.num_labels)
        loss = loss_fct(logits.view(-1, self.model.num_labels),
                        labels.float().view(-1, self.model.num_labels))
        return (loss, outputs) if return_outputs else loss


class BinaryTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        loss_fct = torch.nn.BCEWithLogitsLoss()
        # loss_fct = torch.nn.MultiLabelSoftMarginLoss()
        # print("logits", get_tensor_info(logits))
        # print("self.model.num_labels", self.model.num_labels)
        loss = loss_fct(logits.view(-1),
                        labels.float().view(-1))
        loss = loss_fct(logits.view(-1),
                        labels.float().view(-1))
        return (loss, outputs) if return_outputs else loss


@dataclass
class MetricsComputer:
    tokenizer: PreTrainedTokenizerBase = None
    metric: Any = load_metric("sacrebleu")
    metric2: Any = load_metric("bertscore")
    bart_score_metric: Any = True
    rouge_metric: Any = load_metric("rouge")
    meteor_metric: Any = load_metric("meteor")
    ppl_metric: Any = load_metric("meteor")
    save_dev_location: Any = None
    save_test_location: Any = None
    # no_dl_score: bool = False

    is_script_completion: Any = False
    task_mode: Any = ""
    plm: Any = ""
    input_data: Any = None
    input_file: Any = None

    input_sents: Any = None
    input_text_from_ids: Any = None
    input_nbs: Any = None

    print_input: Any = True

    tgt_text: Any = None

    eval_f: Any = Evaluate()
    model: Any = None
    device: Any = None
    args: Any = None
    is_dev: Any = False
    is_test: Any = False

    def __call__(self, eval_preds):
        print("in metric compute")
        if not self.args.pred_with_gen:
            return {}

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in the labels as we can't decode them.
        # cur_pad_token=self.tokenizer.pad_token
        # if cur_pad_token is None:
        #     cur_pad_token="<|pad|>"
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        cur_eos_token=self.tokenizer.eos_token
        if cur_eos_token is None:
            cur_eos_token=self.tokenizer.sep_token
        cur_bos_token=self.tokenizer.bos_token
        if cur_bos_token is None:
            cur_bos_token=self.tokenizer.cls_token

        if self.is_script_completion or self.task_mode == "prt":
            if "gpt2" in self.plm:
                decoded_labels = self.tgt_text
            else:
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
                decoded_labels = [decoded_label.replace(cur_bos_token, "").replace(self.tokenizer.cls_token, "") \
                                      .replace(self.tokenizer.sep_token, "").replace(cur_eos_token, "").replace(self.tokenizer.pad_token, "") for decoded_label in
                                  decoded_labels]

            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
            if "gpt2" in self.plm:
                ending_pos=[x.find(cur_eos_token) for i, x in enumerate(decoded_preds)]
                # ending_pos=[x if x != -1 else -1 for i, x in enumerate(ending_pos) ]
                decoded_preds = [decoded_pred[len(item):eposs] for decoded_pred,item,eposs in zip(decoded_preds, self.input_text_from_ids, ending_pos)]
            decoded_preds = [decoded_pred.replace(cur_bos_token, "").replace(self.tokenizer.cls_token, "") \
                                 .replace(cur_eos_token, "").replace(cur_eos_token, "").replace(self.tokenizer.pad_token, "") for
                             decoded_pred in decoded_preds]
        else:
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels_special = postprocess_text(decoded_preds, decoded_labels)
        decoded_labels = [sent[0] for sent in decoded_labels_special]

        # print("decoded_preds, decoded_labels", decoded_preds, decoded_labels)

        # Bleu
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels_special)
        result = {"bleu": result["score"]}

        decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels_special))

        # Rouge
        # decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        # result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels))
        rouge = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge = {key: round(value.mid.fmeasure * 100, 2) for key, value in rouge.items()}
        result.update(rouge)

        # Meteor
        meteor = self.meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor["meteor"] *= 100
        result.update(meteor)

        # set
        result["retrieval_score"] = -1
        if self.args.pretrain_concept_generator:
            predicted_set = [[item.strip().lower() for item in p.strip()[1:-1].split(",")] for p in decoded_preds]
            label_set = [[item.strip().lower() for item in p.strip()[1:-1].split(",")] for p in decoded_labels]
            intersection_scores = []
            for p, l in zip(predicted_set, label_set):
                p, l = set(p), set(l)
                intersection_scores.append(len(p.intersection(l)) / len(p.union(l)))
            # intersection_scores=[len(set(p).intersection(set(l)))/len(set(p).union(set(l))) for p,l in zip(predicted_set,label_set)]
            result["retrieval_score"] = np.mean(intersection_scores)

        if self.args.visualize_scatterplot:
            tmp = {}
            prefix_name=f"{self.args.analysis_dir}"+ \
                                   f"{self.args.data_mode}" + \
                                   f"_{self.args.model_name}" + \
                              f"_factorexpander{self.args.factor_expander}" + \
                              f"_topk{self.args.topk_prompt}" + \
                              f"_upc{self.args.use_pretrained_concepts}" + \
                              f"_egold{self.args.eval_gold_concepts}" + \
                              f"_silvereval{self.args.silver_eval}" + \
                          ("_dev" if self.is_dev else "") + \
                          ("_test" if self.is_test else "") + \
                            (f"_top{self.args.top_few}" if int(self.args.top_few)!=-1 else "")
            if "rouge" in self.args.scatterplot_metrics:
                rouges = [self.rouge_metric.compute(predictions=[decoded_pred], references=[decoded_label]) for i, (decoded_pred, decoded_label) in
                          enumerate(tqdm(zip(decoded_preds, decoded_labels)))]
                for j, item in enumerate(tqdm(rouges)):
                    for key, value in item.items():
                        item[key] = round(value.mid.fmeasure * 100, 1)

                tmp["rouges"] = rouges
                dump_file(tmp, f"{prefix_name}_results.json")

            if "bleu" in self.args.scatterplot_metrics:
                bleus = []
                for j, (decoded_pred, decoded_label) in enumerate(tqdm(zip(decoded_preds, decoded_labels_special))):
                    print(decoded_pred, decoded_label)
                    bleus.append(self.metric.compute(predictions=[decoded_pred], references=[decoded_label if decoded_label else "."]))

                tmp["bleus"] = bleus
                # embed()
                BLEUs = []
                # BLEUs = [self.eval_f.evaluate(live=True, cand={j:decoded_preds_alt[j]}, ref=[decoded_label]) for j, decoded_label in tqdm(enumerate(decoded_labels_special), desc="BLEUs")]
                decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
                for j, decoded_label in tqdm(enumerate(decoded_labels_special), desc="BLEUs"):
                    BLEUs.append(self.eval_f.evaluate(live=True, cand={0: decoded_preds_alt[j]}, ref=[decoded_label]))
                tmp["BLEUs"] = BLEUs
                # embed()
                # dump_file(tmp, f"{self.args.analysis_dir}{self.args.model_name}_results.json")
                dump_file(tmp, f"{prefix_name}_results.json")

            if "meteor" in self.args.scatterplot_metrics:
                meteors = [self.meteor_metric.compute(predictions=[decoded_pred], references=[decoded_label]) for decoded_pred, decoded_label in zip(decoded_preds, decoded_labels)]
                for item in meteors:
                    for key, value in item.items():
                        item[key] = value * 100
                tmp["meteors"] = meteors

            # print("tmp", tmp)
            # embed()
            # dump_file(tmp, f"{self.args.analysis_dir}{self.args.model_name}_results.json")
            dump_file(tmp, f"{prefix_name}_results.json")

        # BertScore
        if not self.args.debug and not self.args.no_dl_score:
            self.model.cpu()
            result_bertscore = self.metric2.compute(predictions=decoded_preds, references=decoded_labels, lang="en", batch_size=64)  # device="cpu"
            result["bertscore"] = sum(result_bertscore['f1']) / len(result_bertscore['f1'])
            result["bertscore"] *= 100
            self.model.to(self.device)

            # if self.bart_score_metric is not None:
            self.model.cpu()
            bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
            bart_scorer.load(path='BARTScore/bart.pth')
            result["bartscore"] = np.mean(bart_scorer.score(decoded_preds, decoded_labels, batch_size=16))
            bart_scorer.model.cpu()
            bart_scorer = None
            self.model.to(self.device)
        else:
            result["bertscore"] = -1
            result["bartscore"] = 1

        # result2 = metric2.compute(predictions=decoded_preds, references=[sent[0] for sent in decoded_labels], lang="en", device="cpu")
        # result["bertscore"] = sum(result2['f1']) / len(result2['f1'])

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        result = {k: round(v, 4) for k, v in result.items()}

        if self.print_input:
            assert len(self.input_sents) == len(decoded_preds)
            assert len(self.input_sents) == len(decoded_labels)

            outputs = []
            if self.args.update_input:
                data_file = load_file(f"{self.input_file}")
            for i, (a, b, c, d) in enumerate(zip(self.input_sents, decoded_preds, decoded_labels, self.input_data)):  # , d , self.input_nbs
                # replace(self.tokenizer.sep_token, "\n" + " " * 18).
                # tmp=a.strip().replace(self.tokenizer.cls_token, "").replace(self.tokenizer.pad_token, "")
                # tmp_nb=d.strip().replace(self.tokenizer.cls_token, "").replace(self.tokenizer.pad_token, "")
                # tmp_b=b.strip().replace(self.tokenizer.cls_token, "").replace(self.tokenizer.pad_token, "")
                #
                #
                # a, _=tmp.split(self.tokenizer.sep_token)[0], "\n".join(tmp.split(self.tokenizer.sep_token)[1:])
                # d, _=tmp_nb.split(self.tokenizer.sep_token)[0], "\n".join(tmp_nb.split(self.tokenizer.sep_token)[1:])
                # b, _=tmp_b.split(self.tokenizer.sep_token)[0], "\n".join(tmp_b.split(self.tokenizer.sep_token)[1:])
                cur_steps = modify_output(a, self.tokenizer)
                pred_next_steps = modify_output(b, self.tokenizer, is_tgt=True)
                true_next_steps = c
                true_next_steps = modify_output(c, self.tokenizer, is_tgt=True)

                print("\n\n[Current Steps]: ", cur_steps)
                # print("[Raw Predicted Next Step]: ", b)
                print("[Predicted Next Step]: ", pred_next_steps)
                print("[True Next Step]: ", true_next_steps)
                outputs.append({
                    "cur_steps": a,
                    "pred_next_steps": b,
                    "true_next_steps": c
                })
                if self.args.update_input:
                    data_file[d["sample_id"]]["predicted"] = b
            if self.args.update_input:
                dump_file(data_file, f"{self.input_file}")
            if self.args.save_output:
                dump_file(outputs, f"{self.args.analysis_dir}"+ \
                                   f"{self.args.data_mode}" + \
                                   f"_{self.args.model_name}" + \
                              f"_factorexpander{self.args.factor_expander}" + \
                              f"_topk{self.args.topk_prompt}" + \
                              f"_upc{self.args.use_pretrained_concepts}" + \
                              f"_egold{self.args.eval_gold_concepts}" + \
                              f"_do_cl{self.args.do_cl}" + \
                              f"_sampt{self.args.cl_sample_types}" + \
                              f"_em{self.args.cl_empty_mode}" + \
                              f"_ow{self.args.orig_cl_way}" + \
                          ("_dev" if self.is_dev else "") + \
                          ("_test" if self.is_test else "") + \
                         "_outputs.json")
            if self.save_dev_location:
                dump_file(outputs, self.save_dev_location)
            if self.save_test_location:
                dump_file(outputs, self.save_test_location)

                # print("[Top Neighbor Steps]: ", modify_output(d, self.tokenizer))
                # print("\n\n[Current Steps]: ", a.
                #       replace("[GOAL]", " [GOAL]\n"+ " " * 18).
                #       replace("[SUBGOAL]", " [SUBGOAL]\n"+ " " * 18 ).
                #       replace("[STEP]", " [STEP]\n"+ " " * 18 ).strip()
                #       )
                # print("[Predicted Next Step]: ", b, )
                # print("[True Next Step]: ", c[0], )
                # print("[Top Neighbor Steps]: ", d.
                #       replace("[GOAL]", " [GOAL]\n" + " " * 18).
                #       replace("[SUBGOAL]", " [SUBGOAL]\n" + " " * 18).
                #       replace("[STEP]", " [STEP]\n" + " " * 18).strip()
                #       )
                # print("neighbours")
                # print(nbs)

        else:
            for i, (b, c) in enumerate(zip(decoded_preds, decoded_labels)):
                print("\n[Predicted Next Step]: ", b, )
                print("[True Next Step]: ", c[0], )

        return result


def train_clf_trainer(args, model, optimizer, tokenizer, data, id2label=None, eval_only=False, verbose=False):
    train_data, val_data, test_data = None, None, None
    if not args.eval_only:
        train_data, val_data, test_data = data

        print("\n\nlen(train_data)", len(train_data))
        print("len(val_data)", len(val_data))
        print("len(test_data)", len(test_data))
    else:
        test_data = data
        print("len(test_data)", len(test_data))

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     labels = labels.tolist()
    #     # print("logits, labels ", logits, labels)
    #     logits = (torch.sigmoid(get_tensor_float(logits)) > 0.5).int().tolist()
    #
    #     return get_scores_multilabel_clf(logits, labels)  # todo

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.tolist()
        # print("logits, labels ", logits, labels)
        logits = (torch.sigmoid(get_tensor_float(logits)) > 0.5).view(-1).int().tolist()
        return get_scores_binary_clf(logits, labels)  # todo

    "=========Train========="""
    total_steps = (len(train_data) // args.batch_size + 1) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print("\ntotal_steps", total_steps)
    print("warmup_steps", warmup_steps)
    print("strategy", args.eval_strategy)

    # args.eval_steps = total_steps // args.num_evals
    args.eval_steps = (len(train_data) // args.batch_size + 1) // args.num_evals_per_epoch // (args.true_batch_size // args.batch_size)
    print("eval_steps", args.eval_steps)

    scheduler = get_scheduler(optimizer, args.scheduler, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                              eta_min=0, T_max=(int(args.num_epochs) // 4) + 1)
    print("args.true_batch_size // args.batch_size", args.true_batch_size // args.batch_size)
    training_args = TrainingArguments(
        evaluation_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=bool(args.use_amp),
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay,
        # adam_epsilon=args.adam_epsilon,
        num_train_epochs=args.num_epochs,
        learning_rate=args.plm_lr,
        seed=args.seed,
        load_best_model_at_end=True,
        label_smoothing_factor=args.label_smoothing_factor,
        lr_scheduler_type=SchedulerType(args.scheduler),
        report_to=["wandb"] if not args.debug else [],
        metric_for_best_model=args.metric_for_best_model,
        logging_steps=args.eval_steps,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        # warmup_steps=warmup_steps,
        save_total_limit=2,
        group_by_length=True,  # todo True
        no_cuda=bool(not args.use_gpu),
        greater_is_better=args.greater_is_better,  # todo
        gradient_accumulation_steps=args.true_batch_size // args.batch_size,  # todo
        # debug="underflow_overflow",
        # run_name=args.exp,
        # dataloader_pin_memory=False,
        # do_train=False
    )

    # data_collator = CustomCollator(tokenizer, model=model)
    data_collator = CustomCollator(tokenizer, max_length=args.max_seq_len, has_concepts=args.has_concepts)  # collate_wrapper # todo
    trainer = BinaryTrainer(  # todo
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.train()

    # score_dict, res_out= evaluate_clf(args, model, test_data, data_collator=data_collator, verbose=args.verbose, tokenizer=tokenizer, binary=True, need_label_mapping=False)
    # print("test score_dict", score_dict)
    # if not args.debug:
    #     wandb.log(modify_dict_keys(score_dict, prefix="testfg/"))

    # wandb.log({f"testfg/{k}": v for k, v in score_dict.items()})

    # get best val performance
    best_metric = trainer.state.best_metric
    for hist in trainer.state.log_history[::-1]:
        if f"eval_{args.metric_for_best_model}" in hist and hist[f"eval_{args.metric_for_best_model}"] == best_metric:
            tmp_dict = {k.replace("eval_", "eval/"): v for k, v in hist.items()}
            tmp_dict["train/global_step"] = trainer.state.global_step + 1
            print("best val", tmp_dict)
            if not args.debug:
                wandb.log(tmp_dict)  # , step=trainer.state.global_step+1
            break

    # get test  performance
    torch.save(model.state_dict(), args.model_path)

    score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    print("test score_dict", score_dict)
    if not args.debug:
        wandb.log({k.replace("test_", "test/"): v for k, v in score_dict.items()})
    return score_dict


def eval_func(args, model, test_data, data_collator, tokenizer, binary=True, need_label_mapping=False):
    print("\n\n===Testing===")

    metric = load_metric("sacrebleu")  # get_metric_program("sacrebleu", slow_connection=args.slow_connection)
    metric2 = load_metric("bertscore")

    input_nbs = []
    for item in test_data:
        if 'nbs_input_ids' in item and len(item['nbs_input_ids']) and "cbr" in args.components and "bsl" not in args.model_name:
            input_nbs.append(tokenizer.batch_decode([item['nbs_input_ids'][0]])[0])
        else:
            input_nbs.append("None Used")
    metrics_computer_test = MetricsComputer(tokenizer=tokenizer, metric2=metric2,
                                            input_sents=[item['input_text'] for item in test_data], input_nbs=input_nbs,
                                            is_script_completion=args.script_completion, task_mode=args.task_mode, plm=args.plm, model=model, device=args.device, args=args)

    trainer.compute_metrics = metrics_computer_test
    score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    print("test score_dict", score_dict)
    if not args.debug:
        wandb.log({k.replace("test_", "test/"): v for k, v in score_dict.items()})

    """saving"""
    if not args.debug:
        # latest_state_file=glob.glob(os.path.join(args.model_path, "*.pt"))[-1]
        wandb.save(args.output_dir + "*checkpoint*")


def train(args, model, optimizer, tokenizer, data):
    train_data, val_data, test_data = data

    print("\n\nlen(train_data)", len(train_data))
    print("len(val_data)", len(val_data))
    print("len(test_data)", len(test_data))

    # rouge = datasets.load_metric("rouge")

    # reenable
    # if not args.debug:
    "=========Metric========="""
    metric = load_metric("sacrebleu")  # get_metric_program("sacrebleu", slow_connection=args.slow_connection)
    metric2 = load_metric("bertscore")

    # print("len(train_data.instances)", len(train_data.instances))
    # print("len(val_data.instances)", len(val_data.instances))

    # rouge=None
    # def compute_metrics(pred):
    #
    #     labels_ids = pred.label_ids
    #     pred_ids = pred.predictions
    #
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    #     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #
    #     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    #
    #     return {
    #         "rouge2_precision": round(rouge_output.precision, 4),
    #         "rouge2_recall": round(rouge_output.recall, 4),
    #         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    #     }
    "=========Scheduler========="""
    # breakpoint()
    tmp_batch_size=args.batch_size if int(args.true_batch_size)==-1 else args.true_batch_size
    num_steps_per_epoch = (len(train_data) // (tmp_batch_size * args.my_num_devices) + 1)

    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    # scheduler = get_scheduler(optimizer, args.scheduler, warmup_steps=warmup_steps, total_steps=total_steps)
    args.eval_steps = int(num_steps_per_epoch // args.num_evals_per_epoch)  # // (args.true_batch_size // (args.batch_size*args.num_devices))
    scheduler = get_scheduler(optimizer, args.scheduler, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                              eta_min=0, T_max=(int(args.num_epochs) // 4) + 1)
    # breakpoint()
    "=========Train========="""
    print("\ntotal_steps", total_steps)
    # args.eval_steps = 1000#total_steps // 4
    print("eval_steps", args.eval_steps)
    print("warmup_steps", warmup_steps)
    print("strategy", args.eval_strategy)
    print("args.plm_lr", args.plm_lr)
    print("args.args.eval_batch_siz", args.eval_batch_size)
    """don not use scheduler at start according to exp"""
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=bool(args.pred_with_gen),
        # do_predict=False,
        # do_eval=bool(args.pred_with_gen),
        evaluation_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=bool(args.use_amp),
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay,
        # adam_epsilon=args.adam_epsilon,
        num_train_epochs=args.num_epochs,
        learning_rate=args.plm_lr,
        seed=args.seed,
        load_best_model_at_end=bool(args.pred_with_gen),
        # label_smoothing_factor=args.label_smoothing_factor,
        lr_scheduler_type=SchedulerType(args.scheduler),
        report_to=["wandb"] if not args.debug else [],
        metric_for_best_model=args.metric_for_best_model if bool(args.pred_with_gen) else "loss",
        logging_steps=args.eval_steps,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        group_by_length=True,  # todo True
        no_cuda=bool(not args.use_gpu),
        greater_is_better=bool(args.greater_is_better),  # todo
        gradient_accumulation_steps=args.true_batch_size // args.batch_size,  # todo
        # debug="underflow_overflow",
        # run_name=args.exp,
        # dataloader_pin_memory=False,
        # do_train=False
    )

    data_collator = CustomCollator(tokenizer, model=model, max_length=args.max_seq_len,
                                   has_concepts=args.has_concepts, components=args.components,
                                   model_name=args.model_name, use_special_tag=args.use_special_tag, verbose=args.debug, do_cl=args.do_cl, args=args)
    use_input_text_from_ids='input_text_from_ids' in val_data[0]
    metrics_computer_dev = MetricsComputer(tokenizer=tokenizer, metric2=metric2,
                                           input_sents=[item['src_text'] for item in val_data], input_text_from_ids=[item['input_text_from_ids'] for item in val_data] if use_input_text_from_ids else None, tgt_text=[item['tgt_text'] for item in val_data],
                                           is_script_completion=args.script_completion, task_mode=args.task_mode, plm=args.plm, model=model,
                                           device=args.device, args=args, save_dev_location=args.save_dev_location,
                                           input_data=val_data, input_file=args.dev_file, is_dev=True)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=metrics_computer_dev if args.pred_with_gen else None,  # compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.pred_with_gen else []
    )

    # print("args.model_path", args.model_path)`
    # score_dict = trainer.evaluate(metric_key_prefix="dev", eval_dataset=test_data)  # to get best dev eval metric
    # print("\n\n===Testing Degbuggging===")
    # input_nbs = []
    # metrics_computer_test = MetricsComputer(tokenizer=tokenizer, metric2=metric2,
    #                                         input_sents=[item['input_text'] for item in test_data], input_nbs=input_nbs,
    #                                         is_script_completion=args.script_completion, task_mode=args.task_mode, plm=args.plm, model=model, device=args.device, args=args)
    #
    # trainer.compute_metrics = metrics_computer_test
    # score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    # print("test score_dict", score_dict)
    # print("\n\n===Testing Degbuggging===")

    if not args.eval_only:
        trainer.train()

        # if args.pred_with_gen:
        #     trainer.args.load_best_model_at_end = True
        #     trainer.args.metric_for_best_model = args.metric_for_best_model
        #     trainer.args.load_best_model_at_end = True
        #     trainer.train()

        print("Trained")
        # torch.save(model.state_dict(), args.model_path)
        # trainer.evaluate()  # to get best dev eval metric
        # torch.save(model, "model/states/best_one.pt")

        # all_ckpts=list(glob("model/states/checkpoint-*"))
        # all_ckpts_ids=np.array([int(item.split("checkpoint-")[-1]) for item in all_ckpts])
        # print("all_ckpts_ids",all_ckpts_ids)
        # best_ckpt = all_ckpts[all_ckpts_ids.argsort()[-1]]
        # print("best_ckpt", best_ckpt)
        # model = model.from_pretrained(best_ckpt).to(args.device)

        # model = torch.load("model/states/best_one.pt")

        # test data
        # model.load_state_dict(torch.load(args.model_path))
        # model.eval()
        # # embed()
        # evaluate(args, model, test_data, tokenizer, metric2=metric2)

        # get test  performance
        # torch.save(model.state_dict(), args.model_path)
        if not args.debug:
            import shutil
            torch.save(model.state_dict(),os.path.join(wandb.run.dir,f"{wandb.run.id}.pt"))
            # shutil.copy(args.model_path, os.path.join(wandb.run.dir,f"{wandb.run.id}.pt"))
            wandb.save(os.path.join(wandb.run.dir, f"{wandb.run.id}.pt"))
            # wandb.save(args.model_path)
            # wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))
        else:
            torch.save(model.state_dict(), args.model_path)
        # get best val performance
        gib_metric_names = {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "bleu",
                            "bertscore", "bartscore", "meteor",
                            "rouge1", "rouge2", "rougeL", "rougeLsum",
                            "mrr", "accuracy", "mif1", "maf1", "mirecall", "marecall", "miprecision", "maprecision", "retrieval_score", "loss"}
        lib_metric_names = set()  # "loss"
        general_metric_names = {"eval_gen_len", "eval_steps_per_second", "step", "epoch", "eval_samples_per_second", "eval_runtime", }

        best_metric = trainer.state.best_metric
        tmp_dict = None
        valid_general_metric_names = None
        valid_gib_metric_names = None
        valid_lib_metric_names = None
        for hist in trainer.state.log_history:
            if f"eval_{args.metric_for_best_model}" in hist:
                if tmp_dict is None:
                    tmp_dict = hist.copy()
                else:
                    if valid_general_metric_names is None: valid_general_metric_names = set([m for m in general_metric_names if f"eval_{m}" in hist])
                    if valid_gib_metric_names is None: valid_gib_metric_names = set([m for m in gib_metric_names if f"eval_{m}" in hist])
                    if valid_lib_metric_names is None: valid_lib_metric_names = set([m for m in lib_metric_names if f"eval_{m}" in hist])
                    if hist[f"eval_{args.metric_for_best_model}"] == best_metric:
                        for m in valid_general_metric_names:
                            tmp_dict[m] = hist[m]
                    for m in valid_gib_metric_names:
                        m = f"eval_{m}"
                        tmp_dict[m] = max(tmp_dict[m], hist[m])
                    for m in valid_lib_metric_names:
                        m = f"eval_{m}"
                        tmp_dict[m] = min(tmp_dict[m], hist[m])
        if not tmp_dict:
            print("tmp_dict is None")
            # embed()

        tmp_dict = {k.replace("eval_", "eval/"): v for k, v in tmp_dict.items()} if tmp_dict else {}
        tmp_dict["train/global_step"] = trainer.state.global_step + 1
        print("best val", tmp_dict)
        if not args.debug:
            wandb.log(tmp_dict)  # , step=trainer.state.global_step+1
        # for hist in trainer.state.log_history[::-1]:
        #     if f"eval_{args.metric_for_best_model}" in hist and hist[f"eval_{args.metric_for_best_model}"] == best_metric:
        #         tmp_dict = {k.replace("eval_", "eval/"): v for k, v in hist.items()}
        #         tmp_dict["train/global_step"] = trainer.state.global_step + 1
        #         print("best val", tmp_dict)
        #         if not args.debug:
        #             wandb.log(tmp_dict)  # , step=trainer.state.global_step+1
        #         break

    # clean gpu memory
    # model = model.cpu()
    # time.sleep(5)

    """testing"""
    if not args.pred_with_gen:
        trainer.args.per_device_eval_batch_size = trainer.args.per_device_train_batch_size
        args.pred_with_gen = True
        trainer.args.predict_with_generate = True  #########

    model.eval()

    print("\n\n===Running Dev===")
    if args.eval_only and args.run_dev:
        dev_score_dict = trainer.evaluate(metric_key_prefix="eval", eval_dataset=val_data)  # to get best dev eval metric
        print("dev score_dict", dev_score_dict)
        if not args.debug:
            wandb.log({k.replace("eval_", "eval/"): v for k, v in dev_score_dict.items()})

    if args.no_test:
        return

    print("\n\n===Testing===")
    metrics_computer_test = MetricsComputer(tokenizer=tokenizer, metric2=metric2,
                                            input_sents=[item['src_text'] for item in test_data],  input_text_from_ids=[item['input_text_from_ids'] for item in test_data] if use_input_text_from_ids else None, tgt_text=[item['tgt_text'] for item in test_data],
                                            is_script_completion=args.script_completion, task_mode=args.task_mode, plm=args.plm, model=model, device=args.device, args=args,
                                            save_test_location=args.save_test_location,
                                            input_data=test_data, input_file=args.test_file, is_test=True)

    trainer.compute_metrics = metrics_computer_test

    from time import time
    start_time = time()
    score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    used_time = time() - start_time
    print("test duration", used_time)
    print("test score_dict", score_dict)
    if not args.debug:
        wandb.log({k.replace("test_", "test/"): v for k, v in score_dict.items()})

    """saving"""
    # if not args.debug:
    #     # latest_state_file=glob.glob(os.path.join(args.model_path, "*.pt"))[-1]
    #     wandb.save(args.output_dir+"*checkpoint*")

    #
    # embed()
    # print(wandb.config)
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper,
    #                           drop_last=False, worker_init_fn=seed_worker, num_workers=args.num_workers)
    #
    # # get logger
    # logger = args.logger
    # writer = args.writer
    #
    # train_iterator = range(args.start_epoch, int(args.num_epochs) + args.start_epoch)
    # total_steps = int(len(train_loader) * args.num_epochs)
    # warmup_steps = int(total_steps * args.warmup_ratio)
    #
    # scheduler = None
    # if args.scheduler:
    #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
    #                                                 num_training_steps=total_steps)
    #     # scheduler = STLR(optimizer, num_warmup_steps=warmup_steps,
    #     #                                             num_training_steps=total_steps)
    #
    #     # scheduler = CosineAnnealingLR(optimizer, T_max=(int(args.num_epochs) // 4) + 1, eta_min=0)
    #
    # logger.debug(f"Total steps: {total_steps}")
    # logger.debug(f"Warmup steps: {warmup_steps}")
    #
    # scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    #
    # bad_counter = 0
    # best_val_score = -float("inf")
    # best_epoch = 0
    # t_total = time.time()
    # num_steps = 0
    # logger.debug(f"{len(train_loader)} steps for each epoch")
    # # print("train_iterator",train_iterator)
    # for epoch in train_iterator:
    #     gc.collect()
    #     # logger.debug(f"Epoch {epoch}")
    #     t = time.time()
    #
    #     total_loss = 0
    #     for step, batch in enumerate(train_loader):
    #         # logger.debug(f"Step {step}")
    #         # gc.collect()
    #
    #         num_steps += 1
    #         inputs = batch.to(args.device)
    #
    #         # model learning
    #         model.train()
    #
    #         if args.use_amp:
    #             with torch.cuda.amp.autocast():
    #                 loss = model(inputs, args)
    #             scaler.scale(loss).backward()
    #
    #             if (step + 1) % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
    #                 if args.max_grad_norm > 0:
    #                     scaler.unscale_(optimizer)
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #                 if args.scheduler:
    #                     scheduler.step()
    #                 optimizer.zero_grad()
    #         else:
    #             loss = model(inputs, args)
    #             loss.backward()
    #             if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #                 optimizer.step()
    #                 if args.scheduler:
    #                     scheduler.step()
    #                 optimizer.zero_grad()
    #         total_loss += loss.item()
    #
    #     val_score, output = evaluate(args, model, val_data)
    #
    #     if epoch > args.burn_in:
    #         if val_score >= best_val_score:
    #             best_val_score, best_epoch, bad_counter = val_score, epoch, 0
    #             torch.save({
    #                 'epoch': epoch + 1,
    #                 'num_steps': num_steps + 1,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'best_val_score': best_val_score,
    #             }, args.model_path)
    #         else:
    #             bad_counter += 1
    #         if bad_counter == args.patience:
    #             break
    #
    #     logger.debug(f'Epoch {epoch} | Train Loss {total_loss:.8f} | Val Score {val_score:.4f} | '
    #                  f'Time Passed {time.time() - t:.4f}s')
    #     # embed()
    #     # writer.add_scalar('train', total_loss, epoch)
    #     # writer.add_scalar('val', val_score, epoch)
    #     # wandb.log({'loss_train': loss.data.item(),
    #     #            'val_score': val_score,
    #     #            }, step=num_steps)
    #     # print('Time passed', (time.time() - t))
    # logger.debug("Optimization Finished!")
    # logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # logger.debug('Loading {}th epoch'.format(best_epoch))
    #
    # gc.collect()
    # model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    # test_score, output = evaluate(args, model, test_data)
    #
    # # mkdir("analyze")
    # # dump_file(output, "analyze/output.json")
    #
    # logger.debug(f"Test Score {test_score}")
    # sr_file = args.experiment_path + args.exp + "_result.json"
    # sr = load_file(sr_file) if os.path.exists(sr_file) else []
    # hparam = vars(args)
    #
    # # serialize params
    # for key in hparam:
    #     item = hparam[key]
    #     if not isinstance(item, (float, str, int, complex, list, dict, set, frozenset, bool)):
    #         hparam[key] = str(item)
    # hparam["val_score"] = best_val_score
    # hparam["test_score"] = test_score
    # sr.append(hparam)
    #
    # # Plot lines
    # visualize_plot(y=[[hparam["val_score"] for hparam in sr],
    #                   [hparam["test_score"] for hparam in sr]],
    #                name=["val", "test"],
    #                path=args.experiment_path + args.exp + "_result.png")
    #
    # # print("e1")
    # dump_file(sr, sr_file)


def train_clf(args, model, optimizer, tokenizer, data):
    train_data, val_data, test_data = data

    # if args.debug:
    #     # torch.autograd.set_detect_anomaly(True)
    #     num_samples_test=20
    #     train_data.instances = train_data.instances[:num_samples_test]
    #     val_data.instances = val_data.instances[:num_samples_test]
    #     test_data.instances = test_data.instances[:num_samples_test]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wrapper,
                              drop_last=False, worker_init_fn=seed_worker, num_workers=args.num_workers)

    # get logger
    logger = args.logger
    writer = args.writer

    train_iterator = range(args.start_epoch, int(args.num_epochs) + args.start_epoch)
    total_steps = int(len(train_loader) * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = None
    if args.scheduler:
        scheduler = get_scheduler(optimizer, args.scheduler, warmup_steps=warmup_steps, total_steps=total_steps)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=total_steps)
        # scheduler = STLR(optimizer, num_warmup_steps=warmup_steps,
        #                                             num_training_steps=total_steps)

        # scheduler = CosineAnnealingLR(optimizer, T_max=(int(args.num_epochs) // 4) + 1, eta_min=0)

    logger.debug(f"Total steps: {total_steps}")
    logger.debug(f"Warmup steps: {warmup_steps}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    bad_counter = 0
    best_val_score = -float("inf")
    best_epoch = 0
    t_total = time.time()
    num_steps = 0
    logger.debug(f"{len(train_loader)} steps for each epoch")
    # print("train_iterator",train_iterator)
    for epoch in train_iterator:
        gc.collect()
        # logger.debug(f"Epoch {epoch}")
        t = time.time()

        total_loss = 0
        for step, batch in enumerate(train_loader):
            logger.debug(f"Step {step}")
            # gc.collect()

            num_steps += 1
            inputs = batch.to(args.device)

            # model learning
            model.train()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    loss = model(inputs, args)
                scaler.scale(loss).backward()

                if ((step + 1) % args.grad_accumulation_steps) == 0 or step == len(train_loader) - 1:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if args.scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                loss = model(inputs, args)
                loss.backward()
                if step % args.grad_accumulation_steps == 0 or step == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
            total_loss += loss.item()

            if not (num_steps + 1) % args.eval_steps:
                val_score, output = evaluate_clf(args, model, val_data)

                if epoch > args.burn_in:
                    if val_score >= best_val_score:
                        best_val_score, best_epoch, bad_counter = val_score, epoch, 0
                        torch.save({
                            'epoch': epoch + 1,
                            'num_steps': num_steps + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_val_score': best_val_score,
                        }, args.model_path)
                wandb.log({"epoch": epoch,
                           "step": num_steps,
                           "total_loss": total_loss,
                           "val_score": val_score, })

        val_score, output = evaluate_clf(args, model, val_data)

        if epoch > args.burn_in:
            if val_score >= best_val_score:
                best_val_score, best_epoch, bad_counter = val_score, epoch, 0
                torch.save({
                    'epoch': epoch + 1,
                    'num_steps': num_steps + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_score': best_val_score,
                }, args.model_path)
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        logger.debug(f'Epoch {epoch} | Train Loss {total_loss:.8f} | Val Score {val_score:.4f} | '
                     f'Time Passed {time.time() - t:.4f}s')
        # embed()
        # writer.add_scalar('train', total_loss, epoch)
        # writer.add_scalar('val', val_score, epoch)
        # wandb.log({'loss_train': loss.data.item(),
        #            'val_score': val_score,
        #            }, step=num_steps)
        # print('Time passed', (time.time() - t))
    logger.debug("Optimization Finished!")
    logger.debug("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.debug('Loading {}th epoch'.format(best_epoch))

    gc.collect()
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    evaluate_clf(args, model, test_data, split="test")
    # test_score, output =
    # mkdir("analyze")
    # dump_file(output, "analyze/output.json")

    # logger.debug(f"Test Score {test_score}")
    # sr_file = args.experiment_path + args.exp + "_result.json"
    # sr = load_file(sr_file) if os.path.exists(sr_file) else []
    # hparam = vars(args)
    #
    # # serialize params
    # for key in hparam:
    #     item=hparam[key]
    #     if not isinstance(item, (float, str, int, complex, list, dict, set, frozenset, bool)):
    #         hparam[key]=str(item)
    # hparam["val_score"] = best_val_score
    # hparam["test_score"] = test_score
    # sr.append(hparam)
    #
    # # Plot lines
    # visualize_plot(y=[[hparam["val_score"] for hparam in sr],
    #                   [hparam["test_score"] for hparam in sr]],
    #                name=["val", "test"],
    #                 path=args.experiment_path + args.exp + "_result.png")
    #
    # # print("e1")
    # dump_file(sr, sr_file)
