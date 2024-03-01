
tasks="10-19"
objective="extremile"
dataset="concrete"

python scripts/lbfgs.py --dataset $dataset --objective $objective
for optim in sgd srda lsvrg saddlesaga prospect
do
    taskset -c $tasks python scripts/train.py --dataset $dataset --objective $objective --optimizer $optim --n_jobs 8
done