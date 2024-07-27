declare -x MODEL_NAME="runwayml/stable-diffusion-v1-5"
declare -x dataset_name="salmonhumorous/logo-blip-caption"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=128 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 --enable_xformers_memory_efficient_attention --use_8bit_adam
