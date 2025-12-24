#!/bin/bash -l

##############################
#       Job blueprint        #
##############################


#SBATCH --time=2-4:00:00
#SBATCH --account=cerebras   # Specify your Slurm account here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=helmet_short
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --mail-user=abhishek.maiti@mbzuai.ac.ae
#SBATCH --mail-type=END
#SBATCH --output=slurm_outputs/slurm-%j-%x.out
#SBATCH --error=slurm_outputs/slurm-%j-%x.err 

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Array Job ID                   = $SLURM_ARRAY_JOB_ID"
echo "Array Task ID                  = $SLURM_ARRAY_TASK_ID"
echo "Cache                          = $TRANSFORMERS_CACHE"

source /home/abhishek.maiti/venvs/helmet/bin/activate

export HF_HOME="/lustre/scratch/users/abhishek.maiti/hf_cache"
export OPENAI_API_KEY="xxx"

PORT=$(shuf -i 30000-65000 -n 1)
echo "Port                          = $PORT"

export OMP_NUM_THREADS=8

TAG=v1

CONFIGS=(recall.yaml rag.yaml longqa.yaml summ.yaml icl.yaml rerank.yaml cite.yaml)
#CONFIGS=(${CONFIGS[8]})
SEED=42
RESULTS_DIR="/lustre/scratch/users/abhishek.maiti/HELMET_results"




MODEL_NAME="/lustre/scratch/users/abhishek.maiti/hf_ckpts/LC_8B_SFT_baseline_251113_reasoning_added_9212_128K/checkpoint_9212_to_hf" # CHANGE PATH HERE or you can change the array to load from HF
MNAME=LC_8B_SFT_baseline_251113_reasoning_added_9212_128K
OUTPUT_DIR="$RESULTS_DIR/$MNAME"

OPTIONS=""
shopt -s nocasematch
echo $MNAME


echo "Evaluation output dir         = $OUTPUT_DIR"
echo "Tag                           = $TAG"
echo "Model name                    = $MODEL_NAME"
echo "Options                       = $OPTIONS"

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
python scripts/eval_gpt4_summ.py --output_path $RESULTS_DIR --model_to_check $MNAME &

bash scripts/run_alce.sh $OUTPUT_DIR &
wait;

echo "finished with $?"

echo "All jobs completed successfully"