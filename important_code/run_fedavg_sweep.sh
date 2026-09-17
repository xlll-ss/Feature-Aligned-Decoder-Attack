for local_steps in 1 2 10
do
  for condition in hsic0_control hsic_defended
  do
    if [ "$condition" = "hsic0_control" ]
    then
      target_weight="$CONTROL_CHECKPOINT"
      decoder_prefix="models/decoder_celeba_vgg_undef_clean64_formal_seed"
    else
      target_weight="$DEFENDED_CHECKPOINT"
      decoder_prefix="models/decoder_celeba_vgg_bido_clean64_formal_seed"
    fi
    for decoder_seed in 2027 2028 2029
    do
      decoder_path="${decoder_prefix}${decoder_seed}.pth"
      run_name="${condition}_c4_k8_s${local_steps}_trainbn_decoder${decoder_seed}"
      if [ ! -f "$decoder_path" ]
      then
        echo "Missing decoder: $decoder_path" >&2
        exit 1
      fi
      echo "============================================================"
      echo "condition=${condition}"
      echo "decoder_seed=${decoder_seed}"
      echo "clients=4, client_batch=8, local_steps=${local_steps}"
      echo "observation=model_delta, BN=train"
      echo "============================================================"
      python attack_batch_candidates.py \
        --dataset celeba \
        --data_root "$CELEBA_IMAGES" \
        --target_weight "$target_weight" \
        --decoder_path "$decoder_path" \
        --output_dir "$ROOT/$run_name" \
        --arch vgg \
        --img_size 64 \
        --num_classes 0 \
        --feature_dim 2048 \
        --batch_size 8 \
        --fedavg_clients 4 \
        --client_batch_size 8 \
        --num_batches 25 \
        --num_workers 8 \
        --label_source dataset \
        --observation model_delta \
        --local_steps "$local_steps" \
        --local_lr 0.01 \
        --local_momentum 0 \
        --local_weight_decay 0 \
        --bn_mode train \
        --manifest_path "$MANIFEST" \
        --manifest_partition test \
        --seed 2027 \
        --save_pairs 40 \
        2>&1 | tee "$LOG_ROOT/${run_name}.log"
    done
  done
done
