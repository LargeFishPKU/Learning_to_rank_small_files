work_path=$(pwd)
partition=${1}

job_name='fsl_train'

srun --mpi=pmi2 -p ${partition} --gres=gpu:1 --job-name=${job_name} \
    python -u prank_train.py
