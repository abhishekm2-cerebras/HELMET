#!/bin/bash -l

#SBATCH --time=2-4:00:00
#SBATCH --account=cerebras
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=helmet_64k_sft
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --mail-user=abhishek.maiti@mbzuai.ac.ae
#SBATCH --mail-type=END
#SBATCH --output=/home/abhishek.maiti/projects/abhishekm2-cerebras/jais-family-evals/slurm_outputs/slurm-%j-%x.out
#SBATCH --error=/home/abhishek.maiti/projects/abhishekm2-cerebras/jais-family-evals/slurm_outputs/slurm-%j-%x.err

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

source /home/abhishek.maiti/venvs/helmet/bin/activate

source .env

export HF_HOME="/lustre/scratch/users/abhishek.maiti/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENFORCE_EAGER=1

export OMP_NUM_THREADS=8

# NCCL settings for cross-node communication
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=INFO

TP_SIZE=8  # 2 nodes x 4 GPUs

############################################
#       Ray cluster setup                  #
############################################

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
RAY_PORT=6379

echo "Head node          = $head_node"
echo "Head node IP       = $head_node_ip"
echo "Ray port           = $RAY_PORT"

cleanup() {
    echo "Shutting down Ray cluster..."
    ray stop --force 2>/dev/null
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
        srun --nodes=1 --ntasks=1 -w "$node" ray stop --force 2>/dev/null &
    done
    wait
    echo "Ray cluster shut down."
}
trap cleanup EXIT

echo "Starting Ray head on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --port=$RAY_PORT --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block &
sleep 15

worker_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)
for node in $worker_nodes; do
    echo "Starting Ray worker on $node..."
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$head_node_ip:$RAY_PORT" --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block &
    sleep 5
done

sleep 15
echo "Ray cluster ready."

############################################
#       Evaluation                         #
############################################

TAG=v1

CONFIGS=(ruler_64k.yaml longqa_64k.yaml rag_64k.yaml)
SEED=42
RESULTS_DIR="/lustre/scratch/users/abhishek.maiti/HELMET_results"

MODEL_NAME="/lustre/scratch/users/abhishek.maiti/hf_ckpts/k2v2-instruct-jais-sft_64K"
MNAME="k2v2-jais2-sft-64K-only"
OUTPUT_DIR="$RESULTS_DIR/$MNAME"

OPTIONS="--use_vllm --tensor_parallel_size $TP_SIZE"

echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Tag                           = $TAG"
echo "Model name                    = $MODEL_NAME"
echo "Options                       = $OPTIONS"
echo "Tensor parallel size          = $TP_SIZE"

for CONFIG in "${CONFIGS[@]}"; do
    echo "Config file: $CONFIG"

    python eval.py \
        --config configs/$CONFIG \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --tag $TAG \
        --model_name_or_path $MODEL_NAME \
        --data_root /lustre/scratch/users/abhishek.maiti/HELMET_data \
        $OPTIONS
done

echo "finished with $?"

wait;

python scripts/eval_gpt4_longqa.py --output_path $RESULTS_DIR --model_to_check $MNAME &
python scripts/eval_gpt4_summ.py --output_path $RESULTS_DIR --model_to_check $MNAME --data_root /lustre/scratch/users/abhishek.maiti/HELMET_data/data &

bash scripts/run_alce.sh $OUTPUT_DIR &
wait;

python scripts/collect_results.py --model $MNAME --output_dir $OUTPUT_DIR --tag $TAG --training_length 65536

echo "finished with $?"

echo "All jobs completed successfully"
