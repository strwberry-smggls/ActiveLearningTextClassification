source ~/aih/venv/arl/bin/activate
export TRANSFORMERS_CACHE=~/aih/cache
export TOKENIZERS_PARALLELISM=false
strat="random"
path = "."
seed=1
python main.py datasets/eurlex/ al --al_sampling_strat $strat --checkpoint_path $path"/"$strat"_run/" --train_epochs 15 --annotation_budget 2000 --random_seed $seed --al_sample_size 50 --data_name eurlex


#strat="alternate_subword_alps"
#python main.py datasets/eurlex/ al --al_sampling_strat "subword" "alps" --checkpoint_path "outputs_warmstart/eurlex/fasttext/"$strat"_run"$seed"/" --train_epochs 15 --annotation_budget 2000 --random_seed $seed --initial_size 100 --al_sample_size 100 --data_name eurlex

#seed=2
#python main.py datasets/eurlex/ al --al_sampling_strat $strat --checkpoint_path "outputs_warmstart/eurlex/fasttext/"$strat"_run"$seed"/" --train_epochs 15 --annotation_budget 2000 --random_seed $seed --initial_size 100 --al_sample_size 100 --data_name eurlex 
#seed=3
#python main.py datasets/eurlex/ al --al_sampling_strat $strat --checkpoint_path "outputs_warmstart/eurlex/fasttext/"$strat"_run"$seed"/" --train_epochs 15 --annotation_budget 2000 --random_seed $seed --initial_size 100 --al_sample_size 100 --data_name eurlex 