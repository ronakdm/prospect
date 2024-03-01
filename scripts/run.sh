
tasks="20-29"
objective="esrm"
dataset="iwildcam"

python scripts/lbfgs.py --dataset $dataset --objective $objective
for optim in sgd srda lsvrg saddlesaga prospect
do
    taskset -c $tasks python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8 --use_hyperparam 1
done