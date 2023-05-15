for _datamode in cv fb food; do

  # method 1 retrieval
  for _topk_prompt in 1 2 3; do
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
      --exp_msg "method1" \
      --data_mode $_datamode \
      --silver_eval 0 \
      --use_pretrained_concepts 0 \
      --factor_expander 1 \
      --train_gold_concepts 0 \
      --eval_gold_concepts 0 \
      --topk_prompt $_topk_prompt \
      --pred_with_gen 1
  done

  # method 2 use output from concept generator
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
    --exp_msg "method12" \
    --data_mode $_datamode \
    --silver_eval 0 \
    --use_pretrained_concepts 1 \
    --factor_expander 1 \
    --train_gold_concepts 0 \
    --eval_gold_concepts 0 \
    --topk_prompt 1 \
    --pred_with_gen 1

  # Gold-Concepts
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
    --exp_msg "method12" \
    --data_mode $_datamode \
    --silver_eval 0 \
    --use_pretrained_concepts 0 \
    --factor_expander 1 \
    --train_gold_concepts 1 \
    --eval_gold_concepts 1 \
    --topk_prompt 1 \
    --pred_with_gen 1

done
