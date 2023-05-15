# Tetris

Source code for the paper [Incorporating Task-Specific Concept Knowledge into Script Learning](https://aclanthology.org/2023.eacl-main.220/) (EACL 2023)

## Environments
- Ubuntu-18.0.4
- Python (3.7)
- Cuda (11.1)

## Installation
Install [Pytorch](https://pytorch.org/) 1.9.0, then run the following in the terminal:
```shell
cd code # get into NS/code/ 
conda create -n tetris python=3.7 -y  # create a new conda environment
conda activate tetris

chmod +x scripts/setup.sh
./scripts/setup.sh
```
Then put `data/` folder from [Google Drive](https://drive.google.com/file/d/1pDyv-me64FnNYqmD5G8OGO2UdF2HQ8FZ/view?usp=sharing) into `code/`


## Note
The running of the system might require [wandb](wandb.ai) account login


## Training
Enter the `code/` directory

To directly run on all data categories without contrastive learning, you can run the following (will take  > 20hrs)
```shell
chmod +x scripts/*
./scripts/run_bsl.sh # for baseline
./scripts/run_ours.sh # for our methods
```


Run one of the following for an individual model:

Change `$_datamode` to one of `cv`, `fb`, or `food`. For the retrieval method, change `$_topk_prompt` to one of `1`, `2`, or `3` for different number of neighbors

Note: To run SOCL, add ``--do_cl 1`` and ``--cl_exp_type a/b`` (a or b), change `--model_name` to `sl`, and change `--lr` and `--num_epochs` accordingly



```shell
# baseline
python main.py \
    --use_cache 0  \
    --plm_lr 3e-5 \
    --lr 3e-5 \
    --plm "bart-base" \
    --batch_size 16 \
    --eval_batch_size 16 \
    --no_dl_score 0 \
    --patience 15 \
    --num_epochs 25 \
    --num_evals_per_epoch 2 \
    --exp_msg "bsl"  \
    --data_mode $_datamode \
    --pred_with_gen 1

# method 1 retrieval (CRA)
python main.py \
  --use_cache 0 \
  --plm_lr 3e-5 \
  --lr 3e-5 \
  --plm "bart-base" \
  --batch_size 16 \
  --eval_batch_size 16 \
  --num_epochs 25 \
  --num_evals_per_epoch 2 \
  --no_dl_score 0 \
  --patience 15 \
  --use_generated_factors 1 \
  --exp_msg "method 1" \
  --data_mode $_datamode \
  --silver_eval 0 \
  --use_pretrained_concepts 0 \
  --factor_expander 1 \
  --train_gold_concepts 0 \
  --eval_gold_concepts 0 \
  --topk_prompt $_topk_prompt \
  --pred_with_gen 1

# method 2 use output from concept generator (CG)
python main.py \
    --use_cache 0 \
    --plm_lr 3e-5 \
    --lr 3e-5 \
    --plm "bart-base" \
    --batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 25 \
    --num_evals_per_epoch 2 \
    --no_dl_score 0 \
    --patience 15 \
    --use_generated_factors 1 \
    --exp_msg "method 2" \
    --data_mode $_datamode \
    --silver_eval 0 \
    --use_pretrained_concepts 1 \
    --factor_expander 1 \
    --train_gold_concepts 0 \
    --eval_gold_concepts 0 \
    --topk_prompt 1 \
    --pred_with_gen 1

# Gold-Concepts Variant
python main.py \
    --use_cache 0 \
    --plm_lr 3e-5 \
    --lr 3e-5 \
    --plm "bart-base" \
    --batch_size 16 \
    --eval_batch_size 16 \
    --num_epochs 25 \
    --num_evals_per_epoch 2 \
    --no_dl_score 0 \
    --patience 15 \
    --use_generated_factors 1 \
    --exp_msg "method gold" \
    --data_mode $_datamode \
    --silver_eval 0 \
    --use_pretrained_concepts 0 \
    --factor_expander 1 \
    --train_gold_concepts 1 \
    --eval_gold_concepts 1 \
    --topk_prompt 1 \
    --pred_with_gen 1
```




