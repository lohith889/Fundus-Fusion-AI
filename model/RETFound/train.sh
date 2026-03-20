# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_mae"

# ==== Data settings ====
DATASET="dataset"
NUM_CLASS=2
data_path="./${DATASET}"
task="glaucoma"

# ==== Download pre-trained weights if not present ====
if [ ! -f "./weights/RETFound_mae_natureCFP.pth" ]; then
  echo "Downloading pre-trained RETFound weights..."
  python download_weights.py
fi

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --finetune "./weights/RETFound_mae_natureCFP.pth" \
  --savemodel \
  --global_pool \
  --batch_size 4 \
  --world_size 1 \
  --epochs 10 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${data_path}" \
  --input_size 256 \
  --task "${task}" \
  --adaptation "${ADAPTATION}"
