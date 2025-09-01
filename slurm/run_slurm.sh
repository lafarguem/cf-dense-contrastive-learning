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
PARTITION="gpus48"
GPU=1
NODES=1
TIME=""
LOG_DIR=""
EXP_NAME=""
ROOT_DIR="/vol/biomedic3/bglocker/mscproj/mal224/DCCL"
ENV_PATH="/vol/biomedic3/bglocker/mscproj/mal224/miniforge3/bin/activate cf-contrastive-seg"
JOB_NAME=""
NODELIST=""
EXCLUDE=""
DEPENDENCY=""
COMMAND=""
TIMESTAMP=""
DRY_RUN=false
MASTER_PORT=${MASTER_PORT:-$((12000 + RANDOM % 10000))}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --partition) PARTITION="$2"; shift ;;
        --gpu) GPU="$2"; shift ;;
        --nodes) NODES="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --log_dir) LOG_DIR="$2"; shift ;;
        --root_dir) ROOT_DIR="$2"; shift ;;
        --command) COMMAND="$2"; shift ;;
        --exp_name) EXP_NAME="$2"; shift ;;
        --job_name) JOB_NAME="$2"; shift ;;
        --nodelist) NODELIST="$2"; shift ;;
        --exclude) EXCLUDE="$2"; shift ;;
        --dry-run) DRY_RUN=true ;;
        --dependency) DEPENDENCY="$2"; shift ;;
        --timestamp) TIMESTAMP="$2"; shift;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Ensure --command is provided
if [ -z "$COMMAND" ]; then
    echo "Error: --command 'python -m ...' is required"
    exit 1
fi

# Distributed if more than 1 GPU or more than 1 node
IS_DISTRIBUTED=false
if [ "$GPU" -gt 1 ] || [ "$NODES" -gt 1 ]; then
    IS_DISTRIBUTED=true
fi

if [ -z "$EXP_NAME" ]; then
    exp_name=$(echo "$COMMAND" | grep -oP '\+experiment=\K[^ ]+')
    if [[ -z "$exp_name" ]]; then
        EXP_NAME="default"
    else
        EXP_NAME=$exp_name
    fi
fi

if [ -z "$LOG_DIR" ]; then
    extracted=${COMMAND#*-m }       # remove everything up to "-m "
    log_dir=${extracted%%.*}  # remove everything after the first "."
    if [[ -z "$log_dir" ]]; then
        LOG_DIR="pretraining"
    else
        LOG_DIR=$log_dir
    fi
fi

if [ -z "$JOB_NAME" ]; then
    job_name="${LOG_DIR:0:1}_"
    for part in ${EXP_NAME//_/ }; do
        job_name+=${part:0:1}
    done
    JOB_NAME=$job_name
fi

TIMESTAMP=${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}

# Setup log/output directories
HYDRA_DIR="outputs/${LOG_DIR}/${EXP_NAME}/${TIMESTAMP}"
SLURM_LOG_DIR="${HYDRA_DIR}/slurm"

# Normalize paths
ROOT_DIR_ABS=$(realpath "$ROOT_DIR")
SLURM_LOG_DIR_ABS=$(realpath -m "${ROOT_DIR_ABS}/${SLURM_LOG_DIR}")
HYDRA_DIR_ABS=$(realpath -m "${ROOT_DIR_ABS}/${HYDRA_DIR}")
PYTHONPATH_VAL="$ROOT_DIR_ABS"

# Create log dir
mkdir -p "$SLURM_LOG_DIR_ABS"
echo "$FULL_COMMAND" > "${SLURM_LOG_DIR_ABS}/launch_command.txt"
SLURM_SCRIPT="${SLURM_LOG_DIR_ABS}/slurm_command.sh"

# Generate SLURM script
{
echo "#!/bin/bash"
echo "#SBATCH --partition=${PARTITION}"
echo "#SBATCH --nodes=${NODES}"
echo "#SBATCH --gres=gpu:${GPU}"
echo "#SBATCH --output=${SLURM_LOG_DIR_ABS}/${EXP_NAME}.%N.%j.log"
[ -n "$TIME" ] && echo "#SBATCH --time=${TIME}"
[ -n "$JOB_NAME" ] && echo "#SBATCH --job-name=${JOB_NAME}"
[ -n "$NODELIST" ] && echo "#SBATCH --nodelist=${NODELIST}"
[ -n "$EXCLUDE" ] && echo "#SBATCH --exclude=${EXCLUDE}"
[ -n "$DEPENDENCY" ] && echo "#SBATCH --dependency=$DEPENDENCY"

$IS_DISTRIBUTED && echo "#SBATCH --signal=USR1@60"

echo ""
echo "source ${ENV_PATH}"
echo "export PYTHONPATH=${PYTHONPATH_VAL}"
echo "cd ${ROOT_DIR_ABS}"
echo "export OUTPUT_DIR=${HYDRA_DIR_ABS}"
$IS_DISTRIBUTED && echo "export NCCL_P2P_LEVEL=LOC"
echo ""

if $IS_DISTRIBUTED; then
    echo "MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)"
    echo "echo \"MASTER_ADDR=\$MASTER_ADDR\""
    echo "echo \"SLURM_JOB_NUM_NODES=\$SLURM_JOB_NUM_NODES\""
    echo "echo \"SLURM_NODEID=\$SLURM_NODEID\""
    echo ""
    echo "torchrun \\"
    echo "  --nproc_per_node=${GPU} \\"
    echo "  --nnodes=\$SLURM_JOB_NUM_NODES \\"
    echo "  --node_rank=\$SLURM_NODEID \\"
    echo "  --master_addr=\$MASTER_ADDR \\"
    echo "  --master_port=${MASTER_PORT} \\"
    echo "  ${COMMAND} train.exp_name=${EXP_NAME}"
    echo ""
    echo "echo \"torchrun exited with code \$?\""
else
    echo "python ${COMMAND} train.exp_name=${EXP_NAME}"
fi

echo ""
echo "unset OUTPUT_DIR"
} > "$SLURM_SCRIPT"

# Run or dry run
if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN: SLURM script ==="
    cat "$SLURM_SCRIPT"
    echo "=== End of SLURM script ==="
else
    sbatch "$SLURM_SCRIPT"
fi
