#!/bin/bash
quote_args() {
    local quoted=()
    for arg in "$@"; do
        # Escape and wrap only if necessary
        if [[ "$arg" =~ [^a-zA-Z0-9._:/=+-] ]]; then
            quoted+=("$(printf "'%s'" "$(printf "%s" "$arg" | sed "s/'/'\\\\''/g")")")
        else
            quoted+=("$arg")
        fi
    done
    echo "${quoted[@]}"
}

FULL_COMMAND="bash $0 $(quote_args "$@")"

# Default values
EXP_NAME="default"
ROOT_DIR="/vol/biomedic3/bglocker/mscproj/mal224/DCCL"
PRETRAINING_SLURM_ARGS="--job_name PT --gpu 1 --partition gpus48"
TRANSFER_SLURM_ARGS="--job_name TF --gpu 1 --partition gpus24"
PRETRAINING_OVERRIDES=""
TRANSFER_OVERRIDES=""
TIMESTAMP=""
CKPT="best.pt"

print_usage() {
  echo "Usage: $0 --exp_name <name> --pretraining_slurm_args '<args>' --transfer_slurm_args '<args>' --pretraining_overrides '<cmd>' --transfer_overrides '<cmd>'"
  echo
  echo "Example:"
  echo "$0 --exp_name myexp \\"
  echo "   --pretraining_slurm_args \"--gpu 4 --nodes 1 --partition=gpus24\" \\"
  echo "   --transfer_slurm_args \"--gpu 2 --nodes 1 --partition=gpus8\" \\"
  echo "   --pretraining_overrides \"+experiment=pretraining\" \\"
  echo "   --transfer_overrides \"+experiment=transfer\""
}

# Parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --exp_name)
      EXP_NAME="$2"
      shift; shift
      ;;
    --root_dir)
      ROOT_DIR="$2"
      shift; shift
      ;;
    --pretraining_slurm_args)
      PRETRAINING_SLURM_ARGS="$2"
      shift; shift
      ;;
    --transfer_slurm_args)
      TRANSFER_SLURM_ARGS="$2"
      shift; shift
      ;;
    --pretraining_overrides)
      PRETRAINING_OVERRIDES="$2"
      shift; shift
      ;;
    --transfer_overrides)
      TRANSFER_OVERRIDES="$2"
      shift; shift
      ;;
    --best)
      CKPT="best.pt"
      shift
      ;;
    --last)
      CKPT="last.pt"
      shift
      ;;
    --timestamp) 
      TIMESTAMP="$2"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

TIMESTAMP=${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}

FULL_DIR="outputs/full/${EXP_NAME}/${TIMESTAMP}"
mkdir -p "$FULL_DIR"

PRETRAINING_CMD="-m pretraining.main $PRETRAINING_OVERRIDES runtime.pipeline_metadata='$ROOT_DIR/$FULL_DIR/metadata.json'"

echo "Launching pretraining job..."
PRETRAINING_OUTPUT_DIR="pretraining/${EXP_NAME}/${TIMESTAMP}"
PRETRAINING_JOB_ID=$(bash ./DCCL/slurm/run_slurm.sh \
    --exp_name "$EXP_NAME" \
    --root_dir "$ROOT_DIR" \
    --log_dir "pretraining" \
    --timestamp "$TIMESTAMP" \
    $PRETRAINING_SLURM_ARGS \
    --command "$PRETRAINING_CMD" | awk '{print $4}')

echo "Pretraining job submitted with ID $PRETRAINING_JOB_ID"

TRANSFER_CMD="-m transfer.main $TRANSFER_OVERRIDES train.pretrained_model='$FULL_DIR/pretraining/weights/$CKPT' runtime.pipeline_metadata='$ROOT_DIR/$FULL_DIR/metadata.json' runtime.pretrain_id=$PRETRAINING_JOB_ID"

echo "Launching transfer job with dependency on pretraining..."
TRANSFER_OUTPUT_DIR="transfer/${EXP_NAME}/${TIMESTAMP}"
TRANSFER_JOB_ID=$(bash ./DCCL/slurm/run_slurm.sh \
    --exp_name "$EXP_NAME" \
    --root_dir "$ROOT_DIR" \
    --log_dir "transfer" \
    --timestamp "$TIMESTAMP" \
    --dependency "afterok:${PRETRAINING_JOB_ID}" \
    $TRANSFER_SLURM_ARGS \
    --command "$TRANSFER_CMD" | awk '{print $4}')
echo "Transfer job submitted with ID $TRANSFER_JOB_ID"


# Create symlinks inside full/exp_name/timestamp
cd "$FULL_DIR" || exit 1

mkdir -p "$ROOT_DIR/$FULL_DIR"
ln -s "$ROOT_DIR/outputs/$PRETRAINING_OUTPUT_DIR" "$ROOT_DIR/$FULL_DIR"
mv "$ROOT_DIR/$FULL_DIR/$TIMESTAMP" "$ROOT_DIR/$FULL_DIR/pretraining"

ln -s "$ROOT_DIR/outputs/$TRANSFER_OUTPUT_DIR" "$ROOT_DIR/$FULL_DIR"
mv "$ROOT_DIR/$FULL_DIR/$TIMESTAMP" "$ROOT_DIR/$FULL_DIR/transfer"

# Save metadata
cat <<EOF > "$ROOT_DIR/$FULL_DIR/metadata.json"
{
  "full_command": "$(printf '%s' "$FULL_COMMAND" | sed 's/"/\\"/g')",
  "experiment": "$(printf '%s' "$EXP_NAME" | sed 's/"/\\"/g')",
  "timestamp": "$(printf '%s' "$TIMESTAMP" | sed 's/"/\\"/g')",
  "pretraining": {
    "path": "outputs/$(printf '%s' "$PRETRAINING_OUTPUT_DIR" | sed 's/"/\\"/g')",
    "job_id": $PRETRAINING_JOB_ID,
    "command": "$(printf '%s' "$PRETRAINING_CMD" | sed 's/"/\\"/g')"
  },
  "transfer": {
    "path": "outputs/$(printf '%s' "$TRANSFER_OUTPUT_DIR" | sed 's/"/\\"/g')",
    "job_id": $TRANSFER_JOB_ID,
    "command": "$(printf '%s' "$TRANSFER_CMD" | sed 's/"/\\"/g')"
  }
}
EOF

echo "Full run setup complete at $FULL_DIR"
