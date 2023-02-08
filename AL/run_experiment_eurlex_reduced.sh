source ~/aih/venv/arl/bin/activate
export TRANSFORMERS_CACHE=~/aih/cache
export TOKENIZERS_PARALLELISM=false
seed=1
dataset=datasets/eurlex_reduced/
outname="eurlex_small_als25_b600"
strat="subword"
budget=600
al_sample_size=25
outpath="outputs_reduced/$outname/"$strat"_run"$seed"/"
python main.py $dataset al --al_sampling_strat $strat --checkpoint_path $outpath --train_epochs 15 --annotation_budget $budget --random_seed $seed --initial_size 10 --model_type bert --al_sample_size $al_sample_size --data_name eurlex #--initial_from_index_file ../../DiscreteSAC/results/arxiv_deepq_bs100_small/eval_initial_training_set.csv

strat="alps"
outpath="outputs_reduced/$outname/"$strat"_run"$seed"/"
python main.py $dataset al --al_sampling_strat $strat --checkpoint_path $outpath --train_epochs 15 --annotation_budget $budget --random_seed $seed --initial_size 10 --model_type bert --al_sample_size $al_sample_size --data_name eurlex #--initial_from_index_file ../../DiscreteSAC/results/arxiv_deepq_bs100_small/eval_initial_training_set.csv

strat="dal"
outpath="outputs_reduced/$outname/"$strat"_run"$seed"/"
python main.py $dataset al --al_sampling_strat $strat --checkpoint_path $outpath --train_epochs 15 --annotation_budget $budget --random_seed $seed --initial_size 10 --model_type bert --al_sample_size $al_sample_size --data_name eurlex #--initial_from_index_file ../../DiscreteSAC/results/arxiv_deepq_bs100_small/eval_initial_training_set.csv

strat="cvirs"
outpath="outputs_reduced/$outname/"$strat"_run"$seed"/"
python main.py $dataset al --al_sampling_strat $strat --checkpoint_path $outpath --train_epochs 15 --annotation_budget $budget --random_seed $seed --initial_size 10 --al_sample_size $al_sample_size --data_name eurlex

strat="random"
outpath="outputs_reduced/$outname/"$strat"_run"$seed"/"
python main.py $dataset al --al_sampling_strat $strat --checkpoint_path $outpath --train_epochs 15 --annotation_budget $budget --random_seed $seed --initial_size 10 --model_type bert --al_sample_size $al_sample_size --data_name eurlex #--initial_from_index_file ../../DiscreteSAC/results/arxiv_deepq_bs100_small/eval_initial_training_set.csv