
tasks="0-9"
objective="extremile"
dataset="yacht"

python scripts/lbfgs.py --dataset $dataset --objective $objective
for optim in sgd
do
    taskset -c $tasks python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_epochs 8 --n_jobs 8
done