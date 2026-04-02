#!/bin/bash -l
#SBATCH --time=2-4:00:00
#SBATCH --account=cerebras
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --job-name=rag64k_jais2
#SBATCH --mail-user=abhishek.maiti@mbzuai.ac.ae
#SBATCH --mail-type=END
#SBATCH --output=/home/abhishek.maiti/projects/abhishekm2-cerebras/jais-family-evals/slurm_outputs/slurm-%j-%x.out
#SBATCH --error=/home/abhishek.maiti/projects/abhishekm2-cerebras/jais-family-evals/slurm_outputs/slurm-%j-%x.err

source /home/abhishek.maiti/venvs/helmet/bin/activate
cd /home/abhishek.maiti/projects/abhishekm2-cerebras/jais-family-evals/HELMET
source ../.env

export HF_HOME="/lustre/scratch/users/abhishek.maiti/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENFORCE_EAGER=1
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=INFO

TP_SIZE=8

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
RAY_PORT=6379

cleanup() {
    ray stop --force 2>/dev/null
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
        srun --nodes=1 --ntasks=1 -w "$node" ray stop --force 2>/dev/null &
    done
    wait
}
trap cleanup EXIT

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --port=$RAY_PORT --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block &
sleep 15

for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2); do
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$head_node_ip:$RAY_PORT" --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block &
    sleep 5
done
sleep 15

python eval.py \
    --config configs/rag_64k.yaml \
    --seed 42 \
    --output_dir /lustre/scratch/users/abhishek.maiti/HELMET_results/jais2-70b-rag64k \
    --tag v1 \
    --model_name_or_path /lustre/scratch/users/abhishek.maiti/hf_ckpts/70B_251210_poetry_retraining_NC_V5_excl_poetry_continuation_corruption_DPO_H1_64K \
    --data_root /lustre/scratch/users/abhishek.maiti/HELMET_data \
    --use_vllm --tensor_parallel_size $TP_SIZE

echo "finished with $?"
