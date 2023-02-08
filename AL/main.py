import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from itertools import count
from collections import namedtuple
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import fasttext
import random
from collections import Counter
# from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

# import pytorch_lightning as pl
# from torchmetrics.functional import accuracy, f1, auroc
# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
from al_sampling_agent import SamplingAgent

# from pl_components import *
from transformers_multilabel_cap import *
from transformers_multilabel_cnn import BertCNN
import pytorch_sac_arl as parl
from pytorch_deepQ import Agent


def arl_q(
    sampling_agent: SamplingAgent,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    values,
):
    rl_agent = Agent(
        gamma=0.9,
        epsilon=1.0,
        batch_size=64,
        n_actions=2,
        eps_end=0.01,
        input_dims=[7],
        lr=0.003,
    )
    rl_agent.seed = values.random_seed
    scores = []
    eps_history = []
    max_episodes = 10
    heldout_data = sampling_agent.heldout_data
    # sampling_agent.sample(1, "masked_lm")
    # sampling_agent.sample(1, "subword")

    # For testing only!!!
    sampling_agent.mlm_dict = {}
    sampling_agent.subword_dict = {}
    tokenizer = AutoTokenizer.from_pretrained(values.bert_model)
    for i_episode in range(max_episodes):
        score = 0
        done = False
        last_loss = 0
        sents_added = 0
        sentence_steps = 0
        budget = values.annotation_budget

        batch_rewards = []
        batch_states = []
        batch_new_states = []
        batch_actions = []
        batch_terminal = []

        train_df_new = train_df
        model = TransformersTextClassification(values.bert_model, len(LABEL_COLUMNS))
        res = train_wrapper(
            model,
            train_df_new,
            val_df,
            test_df,
            values,
            ep=-1,
            return_val=True,
            eval_f1=True,
        )
        start_f1 = res[0]
        last_loss = res[1]
        start_index = heldout_data.index[0]
        last_state = []
        # fraction of data used
        last_state.append(len(train_df) / (len(train_df) + len(heldout_data)))
        # average loss of classifier
        last_state.append(last_loss)
        # current episode
        last_state.append((i_episode + 1) / max_episodes)
        # remaining budget
        last_state.append(len(train_df) / values.annotation_budget)
        # mlm score
        try:
            last_state.append(sampling_agent.mlm_dict[start_index]["acc"])
        except:
            last_state.append(0.0)
        # subword score
        try:
            last_state.append(sampling_agent.subword_dict[start_index]["mean"])
        except:
            last_state.append(0.0)
        # entropy
        last_state.extend(
            get_sentence_state(model, tokenizer, heldout_data["comment_text"][0])
        )
        last_state = np.asarray(last_state).astype(float)
        print(last_state)

        running_index = 0
        while not done:
            for index, row in tqdm(heldout_data.iterrows()):
                if running_index == 0:
                    running_index += 1
                    continue
                text = row["comment_text"]
                action = rl_agent.choose_action(last_state)
                batch_states.append(last_state)
                batch_actions.append(action)

                if action == 1:
                    train_df_new = train_df_new.append(row)
                    sents_added += 1
                    budget -= 1

                state_ = []
                state_.append(len(train_df) / (len(train_df) + len(heldout_data)))
                # average loss of classifier
                state_.append(last_loss)
                # current episode
                state_.append((i_episode + 1) / max_episodes)
                # remaining budget
                if budget == 0:
                    state_.append(0)
                else:
                    state_.append(len(train_df) / budget)
                # mlm score
                try:
                    state_.append(sampling_agent.mlm_dict[index]["acc"])
                except:
                    state_.append(0.0)
                # subword score
                try:
                    state_.append(sampling_agent.subword_dict[index]["mean"])
                except:
                    state_.append(0.0)

                state_.extend(
                    get_sentence_state(
                        model, tokenizer, heldout_data["comment_text"][0]
                    )
                )
                batch_new_states.append(state_)

                if budget == 0:
                    print(f"budget exhausted. ending episode after this update")
                    done = True

                batch_terminal.append(done)
                print(sentence_steps)
                if (sentence_steps == 100) or done:
                    # update policy, assume reward for all sentences in batch
                    print(f"remaining budget is {budget}")
                    if sents_added >= 100:
                        res = train_wrapper(
                            model,
                            train_df_new,
                            val_df,
                            test_df,
                            values,
                            i_episode,
                            return_val=True,
                            eval_f1=True,
                        )
                        new_f1 = res[0]
                        last_loss = res[1]
                        sents_added = 0
                    else:
                        res = train_wrapper(
                            model,
                            train_df_new,
                            val_df,
                            test_df,
                            values,
                            i_episode,
                            return_val=True,
                            eval_f1=True,
                            val_only=True,
                        )
                        new_f1 = res[0]
                        last_loss = res[1]
                    if new_f1 > start_f1:
                        reward = 1
                        start_f1 = new_f1
                    elif new_f1 <= start_f1:
                        reward = -1
                    score += reward
                    for i in range(len(batch_states)):
                        rl_agent.store_transition(
                            batch_states[i],
                            batch_actions[i],
                            reward,
                            batch_new_states[i],
                            batch_terminal[i],
                        )
                    rl_agent.learn()
                    batch_states = []
                    batch_new_states = []
                    batch_actions = []
                    batch_terminal = []
                    sentence_steps = 0

                sentence_steps += 1
                running_index += 1
                last_state = state_
                if done:
                    break
        print(f"episode reward: {score}")
        scores.append(score)
        eps_history.append(rl_agent.epsilon)
    print(scores)
    print(eps_history)


def get_sentence_state(model, tokenizer, text):

    model.to(device)
    state = []
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_token_type_ids=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)

    ids = encoding["input_ids"]
    mask = encoding["attention_mask"]
    outputs = torch.sigmoid(model(ids, mask)).flatten().detach().cpu().numpy()

    ent = 0
    for p in outputs:
        ent += p * 1 / np.log(p)
    state.append(ent)
    return state


def arl(
    agent: SamplingAgent,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    values,
):

    datasets = get_datasets_fromdf(train_df, val_df, test_df, "comment_text")
    SavedAction = namedtuple("SavedAction", ["log_prob", "value"])
    running_reward = 10

    tokenizer = AutoTokenizer.from_pretrained(values.bert_model)
    max_episodes = 10
    log_interval = 1
    torch.manual_seed(RANDOM_SEED)
    for i_episode in range(max_episodes):

        # reset environment and episode reward
        ep_reward = 0
        heldout_df = agent.heldout_data
        last_loss = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        policy = parl.Policy()
        model = TransformersTextClassification(
            values.bert_model, datasets["num_classes"]
        )
        model.to(device)
        sents_added = 0
        batch_sents = 0

        policy_optimizer = optim.Adam(model.parameters(), lr=3e-2)
        budget = values.annotation_budget
        for index, row in tqdm(heldout_df.iterrows()):
            # build current state
            state = []
            """
            should we add full sentence tokenization? maybe not at first
            encoding = tokenizer(row["comment_text"],
                                  add_special_tokens=True,
                                  max_length=512,
                                  padding='max_length',
                                  return_token_type_ids=False,
                                  truncation=True,
                                  return_attention_mask=True,
                                  return_tensors='pt').numpy()
            state.extend(encoding)
            """
            # fraction of data used
            state.append(len(train_df) / (len(train_df) + len(heldout_df)))
            # average loss of classifier
            state.append(last_loss)
            # current episode
            state.append((i_episode + 1) / max_episodes)
            # remaining budget
            state.append(len(train_df) / values.annotation_budget)
            state = np.asarray(state)

            # select action from policy
            action = parl.select_action(state, policy, SavedAction)

            # evaluate action
            if action == 1:
                train_df = train_df.append(row)
                sents_added += 1

            if sents_added == values.al_sample_size:
                # get reward
                reward = train_wrapper(
                    model, train_df, val_df, test_df, values, i_episode, return_val=True
                )
                policy.rewards.extend(reward for i in range(batch_sents))
                ep_reward += sum([reward for i in range(batch_sents)])
                budget -= sents_added

                batch_sents = 0
                sents_added = 0

            # state, reward, done, _ = env.step(action)

            # model.rewards.append(reward)
            # ep_reward += reward
            batch_sents += 1

            if budget <= 0:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        parl.finish_episode(policy, policy_optimizer)

        # log results
        if i_episode % log_interval == 0:
            print(
                "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    i_episode, ep_reward, running_reward
                )
            )

        # check if we have "solved" the cart pole problem
        if running_reward > 10000:
            print(
                "Solved! Running reward is now {} and "
                "the last episode runs to {} time steps!".format(
                    running_reward, sents_added
                )
            )
            break


def train_wrapper(
    model,
    train_df,
    val_df,
    test_df,
    values,
    ep=0,
    return_val=False,
    eval_f1=True,
    val_only=False,
):
    train_params = {
        "batch_size": values.batch_size,
        "shuffle": True,
        "num_workers": values.num_workers,
    }
    test_params = {
        "batch_size": values.batch_size,
        "shuffle": False,
        "num_workers": values.num_workers,
    }
    torch.manual_seed(RANDOM_SEED)
    columns = list(train_df.columns)
    columns = columns[columns.index("comment_text") :]
    train_dataset: Dataset = TextDataset(train_df, "comment_text", columns[1:])
    training_loader = DataLoader(
        train_dataset, collate_fn=BertCollator(values.bert_model), **train_params
    )
    val_dataset = TextDataset(val_df, "comment_text", columns[1:])
    validation_loader = DataLoader(
        val_dataset, collate_fn=BertCollator(values.bert_model), **test_params
    )
    test_dataset = TextDataset(test_df, "comment_text", columns[1:])
    test_loader = DataLoader(
        test_dataset, collate_fn=BertCollator(values.bert_model), **test_params
    )

    if val_only:
        f1_score = val_model(model, validation_loader, eval_f1)
        return f1_score

    checkpoint_path = values.checkpoint_path + f"ep{ep}_{len(train_df)}_checkpoint.pt"
    best_model_path = values.checkpoint_path + f"ep{ep}_{len(train_df)}_best-model.pt"
    model.to(device)
    num_training_steps = len(datasets["train"]) // values.batch_size

    num_warmup_steps = 0.06 * num_training_steps
    # get the optimizer and scheduler
    opt_sch = get_optimizer(
        model=model,
        lr=values.learning_rate,
        epochs=values.train_epochs,
        tr_steps=num_training_steps,
        wrm_steps=num_warmup_steps,
    )
    optimizer = opt_sch["optimizer"]
    scheduler = opt_sch["scheduler"]
    # load best model
    # checkpoint_path = values.checkpoint_path
    # best_model_path = values.best_model_path

    print(f"Training with {len(train_df)} sentences")
    f1 = train_model(
        n_epochs=values.train_epochs,
        training_loader=training_loader,
        validation_loader=validation_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path,
        return_val=return_val,
        eval_f1=eval_f1,
    )
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    with open(f"{values.checkpoint_path}ep{ep}_{len(train_df)}_results.txt", "w") as f:
        f.write(f"Val F1 Score (Macro) = {f1}")
        # f.write(f"Test F1 Score (Macro) = {f1_score_macro}")
    return f1


def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b


def sbert_clustering(heldout_df, num_classes):
    sbert_model = SentenceTransformer("all-mpnet-base-v2")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sbert_model.to(device)
    s_batch = []
    encoded = []
    batch_size = 128
    for s in tqdm(heldout_df["comment_text"].tolist()):
        if len(s_batch) < batch_size:
            s_batch.append(s)
        else:
            enc = sbert_model.encode(s_batch)
            encoded.extend(enc)
            s_batch = []
            s_batch.append(s)
    clustering = KMeans(n_clusters=num_classes, random_state=101).fit(encoded)
    # find data point nearest to each cluster center
    distances = {}
    print(f"getting closest sentences to {num_classes} cluster centers")
    for i in tqdm(range(len(encoded))):
        sent = encoded[i]
        sent_index = i
        clust_label = clustering.labels_[i]
        clust_center = clustering.cluster_centers_[clust_label]
        if clust_label not in distances:
            distances[clust_label] = []
        dist = cosine_distances([sent], [clust_center])
        distances[clust_label].append((sent_index, dist))
    final_indices = []
    for d in distances:
        final_indices.append(sorted(distances[d], key=lambda x: x[1])[0][0])
    return heldout_df.iloc[final_indices]


def al_fasttext(agent, train_df, val_df, test_df, values):
    pass


def dataset_to_fasttext(df, outfile="temp.txt"):
    outstr = ""
    for index, row in tqdm(df.iterrows()):
        text = row["comment_text"]
        labels = []
        index = 2
        for l in row[2:]:
            if l == 1:
                labels.append(df.columns[index])
            index += 1
        label_str = ""
        for l in labels:
            label_str += f"__label__{l} "
        if len(labels) == 0:
            print(labels)
            input("...")
        outstr += f"{label_str} {text}\n"
    all_labels = " ".join(f"__label__{df.columns[index]}" for index in range(2,len(row)))
    outstr += f"{all_labels} none"
    outstr.strip("\n")
    with open(outfile, "w") as f:
        f.write(outstr)

def fasttext_micro_f1(path, t, ft_model):
    tp = 0
    fp = 0
    fn = 0
    for row in open(path).read().split("\n"):
        row_split = row.split(" ")
        labels = []
        text = []
        for r in row_split:
            if r.startswith("__label__"):
                labels.append(r)
            else:
                text.append(r)

        pred = ft_model.predict(" ".join(text), k=-1, threshold=t)
        labels_ = list(pred[0])

        for l in labels:
            if l in labels_:
                tp += 1
            else:
                fn += 1
        for l_ in labels_:
            if l_ not in labels:
                fp += 1
    prec = safe_divide(tp, tp + fp)
    rec = safe_divide(tp, tp + fn)
    f1 = safe_divide((prec * rec) * 2, (prec + rec))
    return f1

def fasttext_train_classifier(path, t, ft_model):
    class_eval = {}
    for row in open(path).read().split("\n"):
        row_split = row.split(" ")
        labels = []
        text = []
        for r in row_split:
            if r.startswith("__label__"):
                labels.append(r)
            else:
                text.append(r)
            
        for l in labels: 
            if l not in class_eval:
                class_eval[l] = {"tp":0,"fp":0,"fn":0}

        pred = ft_model.predict(" ".join(text), k=-1, threshold=t)
        labels_ = list(pred[0])
        for l in labels_:
            if l not in class_eval:
                class_eval[l] = {"tp":0,"fp":0,"fn":0}

        for l in labels:
            if l in labels_:
                class_eval[l]["tp"] += 1
            else:
                class_eval[l]["fn"] += 1
        for l_ in labels_:
            if l_ not in labels:
                class_eval[l_]["fp"] += 1
        
    for l in class_eval:
        tp = class_eval[l]["tp"]
        fp = class_eval[l]["fp"]
        fn = class_eval[l]["fn"]

        prec = safe_divide(tp, tp+fp)
        rec = safe_divide(tp, tp+fn)
        f1 = safe_divide(prec*rec*2, (prec+rec))
        class_eval[l]["f1"] = f1
        #print(list(class_eval.items()))
        #input("...")
    return class_eval

def sample_initial_dataset(df:pd.DataFrame, n=10):
    label_names = list(df.columns[2:])
    labels = []

    for index, row in df.iterrows():
        for l in label_names:
            if row[l] == 1:
                labels.append(l)

    new_label_count = {x:n for x in labels}

    new_rows = []
    for index, row in df.iterrows():
        labels = [l for l in label_names if row[l] == 1]
        add = False
        for l in labels:
            if new_label_count[l] > 0:
                add = True
                break
        if add == True:
            new_rows.append(row)
            #print(new_label_count)
            for l in labels:
                new_label_count[l] -= 1

    new_df = pd.DataFrame(new_rows)

    for index, row in new_df.iterrows():
        for l in label_names:
            if row[l] == 1:
                labels.append(l)

    #print(labels)
    c = Counter(labels)
    df = pd.concat([df, new_df]).drop_duplicates(keep=False)
    return df, new_df
    #df.to_csv("../datasets/arXiv_reduced/train.csv", index=False)
    #new_df.to_csv("../datasets/arXiv_reduced/stratified_train.csv", index=False)       
    
def fasttext_find_threshold(train_df, val_df, values, path):
    dataset_to_fasttext(train_df, path)
    thrl = [0.00001, 0.0001, 0.001, 0.01] + list(np.arange(0.05, 1, 0.05))
    best_f1 = 0
    best_t = 0
    
    ft_model = fasttext.train_supervised(
                input=path,
                lr=0.5,
                epoch=25,
                wordNgrams=2,
                bucket=200000,
                dim=50,
                loss="ova",
                seed=values.random_seed
            )
    dataset_to_fasttext(val_df, path)
    for t_ in thrl:
        class_eval = fasttext_train_classifier(path, t_, ft_model)
        macro_f1 = np.mean([x[1]["f1"] for x in list(class_eval.items())])
        #micro_f1 = fasttext_micro_f1(path, threshold, ft_model)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_t = t_
            ft_model.save_model(f"{values.checkpoint_path}_fasttext.bin")
    return best_t, ft_model
        
            

def al(agent, train_df, val_df, test_df, values):
    text_column = "comment_text"
    datasets = get_datasets_fromdf(train_df, val_df, test_df, text_column)
    budget = values.annotation_budget
    sampling_agent = agent
    train_params = {
        "batch_size": values.batch_size,
        "shuffle": True,
        "num_workers": values.num_workers,
    }
    test_params = {
        "batch_size": values.batch_size,
        "shuffle": False,
        "num_workers": values.num_workers,
    }

    BATCH_SIZE = values.batch_size
    EPOCHS = values.train_epochs
    MAX_TOKEN_COUNT = values.max_token_count
    validation_loader = DataLoader(
        datasets["val"], collate_fn=BertCollator(values.bert_model), **test_params
    )
    testing_loader = DataLoader(
        datasets["test"], collate_fn=BertCollator(values.bert_model), **test_params
    )

    # version_str = f"{values.al_sampling_strat}_bs{BATCH_SIZE}_ep{EPOCHS}"

    checkpoint_path = values.checkpoint_path + f"{len(train_df)}_checkpoint.pt"
    best_model_path = values.checkpoint_path + f"{len(train_df)}_best-model.pt"

    """
    if values.al_sampling_strat in ["finetune"]:
        model = AutoModelForMaskedLM.from_pretrained(values.bert_model)
        agent.alps_model = model
        agent.finetune_bert_lm(AutoTokenizer.from_pretrained(values.bert_model))
        torch.manual_seed(RANDOM_SEED)
        model = TransformersTextClassification(
            "helper_models/alps_model", datasets["num_classes"]
        )
    else:
        torch.manual_seed(RANDOM_SEED)
        model = TransformersTextClassification(
            values.bert_model, datasets["num_classes"]
        )

    model.to(device)
    """

    torch.manual_seed(RANDOM_SEED)
    tok = AutoTokenizer.from_pretrained(values.bert_model)
    model = TransformersTextClassification(values.bert_model, datasets["num_classes"], seed=RANDOM_SEED)
    cycle = 0
    print(values.al_sampling_strat)
    if len(values.al_sampling_strat) > 0 and isinstance(values.al_sampling_strat, list):
        random.seed(values.random_seed)
        random.shuffle(values.al_sampling_strat)

    while (budget > 0) and (len(sampling_agent.heldout_data) > 0):
        torch.manual_seed(RANDOM_SEED)
        if values.cold_start:
            model = BertCNN(values.bert_model, datasets["num_classes"])
        model.to(device)

        # Train using a BERT classifier
        if values.model_type == "bert":
            columns = list(train_df.columns)
            columns = columns[columns.index(text_column) :]
            train_dataset: Dataset = TextDataset(train_df, "comment_text", columns[1:])
            training_loader = DataLoader(
                train_dataset,
                collate_fn=BertCollator(values.bert_model),
                **train_params,
            )

            checkpoint_path = values.checkpoint_path + f"{len(train_df)}_checkpoint.pt"
            best_model_path = values.checkpoint_path + f"{len(train_df)}_best-model.pt"

            # compute training and warmup steps needed for the scheduler
            num_training_steps = len(datasets["train"]) // values.batch_size

            num_warmup_steps = 0.06 * num_training_steps
            # get the optimizer and scheduler
            torch.manual_seed(RANDOM_SEED)
            opt_sch = get_optimizer(
                model=model,
                lr=values.learning_rate,
                epochs=values.train_epochs,
                tr_steps=num_training_steps,
                wrm_steps=num_warmup_steps,
            )
            optimizer = opt_sch["optimizer"]
            scheduler = opt_sch["scheduler"]
            # load best model
            # checkpoint_path = values.checkpoint_path
            # best_model_path = values.best_model_path
            print(f"Training with {len(train_df)} sentences")
            train_model(
                n_epochs=values.train_epochs,
                training_loader=training_loader,
                validation_loader=validation_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                best_model_path=best_model_path,
                fast_val=values.fast_val,
            )
            best_ckpt = load_ckp(best_model_path, model, optimizer)
            model = best_ckpt["model"]
            thr_best = best_ckpt["thr"]
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            # compute performance scores on the test set for the best model
            # test_outputs, test_targets = do_prediction(testing_loader, best_mod)

            # compute performance on val set instead
            test_outputs, test_targets = do_prediction(testing_loader, model)
            with open(f"{values.checkpoint_path}{len(train_df)}_preds.p", "wb") as f:
                pickle.dump((test_outputs, test_targets, thr_best), f)
            test_preds = (np.array(test_outputs) >= thr_best).astype(int)
            accuracy = metrics.accuracy_score(test_targets, test_preds)
            f1_score_micro = metrics.f1_score(test_targets, test_preds, average="micro")
            f1_score_macro = metrics.f1_score(test_targets, test_preds, average="macro")

            with open(f"{values.checkpoint_path}{len(train_df)}_results.txt", "w") as f:
                f.write(f"Test F1 Score (Micro) = {f1_score_micro}\n")
                f.write(f"Test F1 Score (Macro) = {f1_score_macro}")
            train_df.to_csv(
                f"{values.checkpoint_path}{len(train_df)}_train.csv", index=False
            )
            print(f"Test Accuracy Score = {accuracy}")
            print(f"Test F1 Score (Micro) = {f1_score_micro}")
            print(f"Test F1 Score (Macro) = {f1_score_macro}")
            #test_outputs, test_targets = do_prediction(testing_loader, model)
            #test_preds = (np.array(test_outputs) >= thr_best).astype(int)
            #accuracy = metrics.accuracy_score(test_targets, test_preds)
            #f1_score_micro = metrics.f1_score(test_targets, test_preds, average="micro")
            #f1_score_macro = metrics.f1_score(test_targets, test_preds, average="macro")

            with open(
                f"{values.checkpoint_path}{len(train_df)}_test_results.txt", "w"
            ) as f:
                f.write(f"Test F1 Score (Micro) = {f1_score_micro}\n")
                f.write(f"Test F1 Score (Macro) = {f1_score_macro}")

            print(f"Test Accuracy Score = {accuracy}")
            print(f"Test F1 Score (Micro) = {f1_score_micro}")
            print(f"Test F1 Score (Macro) = {f1_score_macro}")

        # Train using fasttext, requires many different formats/datasets
        elif values.model_type == "fasttext":
            
            path = values.checkpoint_path + "fasttext_tmp.txt"
            dataset_to_fasttext(train_df, path)

            threshold, ft_model = fasttext_find_threshold(train_df, val_df, values, path)
            ft_model = fasttext.load_model(f"{values.checkpoint_path}_fasttext.bin")
            dataset_to_fasttext(test_df, path)
            class_eval = fasttext_train_classifier(path, threshold, ft_model)
            macro_f1 = np.mean([x[1]["f1"] for x in list(class_eval.items())])
            micro_f1 = fasttext_micro_f1(path, threshold, ft_model)

            with open(f"{values.checkpoint_path}{len(train_df)}_results.txt", "w") as f:
                f.write(f"Test F1 Score (Micro) = {micro_f1}\n")
                f.write(f"Test F1 Score (Macro) = {macro_f1}")
            with open(f"{values.checkpoint_path}{len(train_df)}_class_eval.txt", "w") as fp:
                outscores = []
                #class_eval = env.eval_per_class[-1]
                for x in list(class_eval.items()):
                    outscores.append((x[0],x[1]["f1"]))
                outstr = f"{len(train_df)},"
                for x in sorted(outscores, key=lambda x:x[1]):
                    outstr += f"{x[0]} {x[1]},"
                    outstr.strip(",")
                outstr += "\n"
                fp.write(outstr)
            train_df.to_csv(
                f"{values.checkpoint_path}{len(train_df)}_train.csv", index=False
            )

        # if values.al_sampling_strat == "alps":
        #    model.model.save_pretrained(f"helper_models/alps_model")

        
        if len(values.al_sampling_strat) > 0 and isinstance(values.al_sampling_strat, list):
            strat = values.al_sampling_strat[cycle % len(values.al_sampling_strat)]
        else:
            strat = values.al_sampling_strat
        #print(strat)
        #input("...")
        train_df = agent.sample(
            values.al_sample_size,
            strat,
            model=model,
            tokenizer=values.bert_model,
            values=values,
        )
        budget -= values.al_sample_size

        cycle += 1
    # compute performance scores on the test set for the best model


def eval_model(trainer, validation, tokenizer, output_path, args):
    trained_model = MultiLabelTagger.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        n_classes=len(LABEL_COLUMNS),
        label_columns=LABEL_COLUMNS,
    )
    trained_model.eval()
    trained_model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = trained_model.to(device)

    val_dataset = MultiLabelDataset(
        LABEL_COLUMNS, validation, tokenizer, max_token_len=args.max_token_count
    )

    predictions = []
    labels = []

    for item in tqdm(val_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device),
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())

    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    y_pred = predictions.numpy()
    y_true = labels.numpy()

    # print(y_pred[:10])
    # print(y_true[:10])

    with open(
        f"{trainer.checkpoint_callback.best_model_path.strip('.ckpt')}_predictions.pkl",
        "wb",
    ) as f:
        results = {"pred": y_pred, "gold": y_true}
        pickle.dump(results, f)

    upper, lower = 1, 0

    thrl = (
        [0.00001, 0.0001, 0.001, 0.01]
        + list(np.arange(0.05, 1, 0.05))
        + [0.9, 0.93, 0.97, 0.98]
    )
    # thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    eval_dict = {
        "macro_t": {"t": 0, "macro_f1": 0, "micro_f1": 0},
        "micro_t": {"t": 0, "macro_f1": 0, "micro_f1": 0},
    }
    for threshold in thrl:
        y_pred = np.where(y_pred > threshold, upper, lower)

        # report = classification_report(y_true,
        #                               y_pred,
        #                               target_names=LABEL_COLUMNS,
        #                               zero_division=0)
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=LABEL_COLUMNS,
            zero_division=0,
            output_dict=True,
        )
        # print(report_dict.keys())
        macro = report_dict["macro avg"]["f1-score"]
        micro = report_dict["micro avg"]["f1-score"]
        print(f"t: {threshold}, macro: {macro}, micro: {micro}")
        if macro > eval_dict["macro_t"]["macro_f1"]:
            eval_dict["macro_t"]["macro_f1"] = macro
            eval_dict["macro_t"]["micro_f1"] = micro
            eval_dict["macro_t"]["t"] = threshold
        if micro > eval_dict["micro_t"]["micro_f1"]:
            eval_dict["micro_t"]["micro_f1"] = micro
            eval_dict["micro_t"]["macro_f1"] = macro
            eval_dict["micro_t"]["t"] = threshold

    print(f"writing evaluation to {output_path}")
    with open(f"{output_path}", "w", encoding="utf8") as f:
        output = f"optimized by macro_f1: macro_f1:{eval_dict['macro_t']['macro_f1']}"
        output += f", micro_f1:{eval_dict['macro_t']['micro_f1']}"
        output += f", t:{eval_dict['macro_t']['t']}"
        output += "\n"
        output += f"optimized by micro_f1: macro_f1:{eval_dict['micro_t']['macro_f1']}"
        output += f", micro_f1:{eval_dict['micro_t']['micro_f1']}"
        output += f", t:{eval_dict['micro_t']['t']}"
        f.write(output)

def df_from_index_file(index_file):
    indices = index_file.iloc[0,-1].split(" ")
    print(indices)
    input("...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Active Learning for Multi Label Classification"
    )
    parser.add_argument(
        "data_folder",
        type=str,
        help="Path to folder containing train, val and test data",
    )
    parser.add_argument(
        "mode", type=str, choices=["al", "arl", "arlq", "test"], default="al"
    )
    parser.add_argument("--checkpoint_path", type=str, default="outputs/unnamed/")
    # parser.add_argument("--best_model_path", type=str, default="best_model.pt")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_token_count", type=int, default=512)
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--initial_size", type=int, default=200)
    parser.add_argument("--al_sample_size", type=int, default=200)
    parser.add_argument("--al_sampling_strat", type=str, nargs="+", default="random")
    parser.add_argument("--annotation_budget", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument("--fast_val", type=bool, default=False)
    parser.add_argument("--initial_clustering", action="store_true")
    parser.add_argument("--cold_start", action="store_true")
    parser.add_argument("--data_name", type=str, default=None)
    parser.add_argument("--use_stratified", action="store_true")
    parser.add_argument(
        "--model_type", type=str, default="bert", choices=["bert", "fasttext"]
    )
    parser.add_argument("--initial_from_index_file", type=str, default="")
    values = parser.parse_args()

    global RANDOM_SEED
    # pl.seed_everything(values.random_seed * 12)
    RANDOM_SEED = values.random_seed * 12

    if values.checkpoint_path == None:
        values.checkpoint_path = values.data_folder

    # load datasets and create train set of initial size
    if values.use_stratified:
        heldout_df = pd.read_csv(f"{values.data_folder}stratified_train.csv")
    else:
        heldout_df = pd.read_csv(f"{values.data_folder}train.csv")
    if not os.path.exists(f"{values.data_folder}dev.csv"):
        heldout_df, val_df = train_test_split(
            heldout_df, test_size=0.2, random_state=RANDOM_SEED
        )
    else:
        val_df = pd.read_csv(f"{values.data_folder}dev.csv")
    if not os.path.exists(f"{values.data_folder}test.csv"):
        heldout_df, test_df = train_test_split(
            heldout_df, test_size=0.2, random_state=RANDOM_SEED
        )
    else:
        test_df = pd.read_csv(f"{values.data_folder}test.csv")
    
    if values.initial_from_index_file != "":
        train_df = pd.read_csv(values.initial_from_index_file)
        heldout_df = pd.concat([train_df, heldout_df]).drop_duplicates(keep=False)
    else:
        heldout_df, train_df = sample_initial_dataset(heldout_df, n=values.initial_size)
    #train_df = heldout_df.sample(n=values.initial_size, random_state=RANDOM_SEED)
    #heldout_df = pd.concat([heldout_df, train_df]).drop_duplicates(keep=False)

    datasets = get_datasets_fromdf(train_df, val_df, test_df, "comment_text")
    if values.initial_clustering:
        print(f"using clustering for initial dataset, ignoring --initial_size")
        train_df = sbert_clustering(heldout_df, datasets["num_classes"])

    if values.fast_val:
        val_df = val_df.sample(n=1000, random_state=RANDOM_SEED)
        heldout_df = heldout_df.head(160)
        print(f"CAUTION! Using small dev set of size {len(val_df)} for rapid testing")
    # TEST CODE ONLY!!!
    # heldout_df = heldout_df.head(100)

    sampling_agent = SamplingAgent(heldout_df, train_df, device=device)
    sampling_agent.data_name = values.data_name
    if not os.path.exists(values.checkpoint_path):
        os.makedirs(values.checkpoint_path)

    global LABEL_COLUMNS
    LABEL_COLUMNS = heldout_df.columns.to_list()[2:]

    if values.mode == "al":
        al(sampling_agent, train_df, val_df, test_df, values)
    elif values.mode == "arl":
        arl(sampling_agent, train_df, val_df, test_df, values)
    elif values.mode == "arlq":
        arl_q(sampling_agent, train_df, val_df, test_df, values)
    elif values.mode == "test":
        #path = values.checkpoint_path + "fasttext_tmp.txt"
        #train_df = pd.read_csv((f"{values.data_folder}train.csv"))
        #train_df = pd.read_csv("../../DiscreteSAC/results/arxiv_deepq_bs100_small/eval_initial_training_set.csv")
        #threshold, ft_model = fasttext_find_threshold(train_df,val_df,values,path)
        #micro_f1 = fasttext_micro_f1(path, threshold, ft_model)
        #class_eval = fasttext_train_classifier(path, threshold, ft_model)
        #macro_f1 = np.mean([x[1]["f1"] for x in list(class_eval.items())])
        #print(micro_f1, macro_f1)
        #sampling_agent.heldout_data = sampling_agent.heldout_data.sample(n=500)
        #sampling_agent.sample(100, "cvirs", model=None, tokenizer=None, values=values)
        
        #sample_initial_dataset(pd.read_csv(f"{values.data_folder}stratified_train.csv"))
        # sampling_agent.model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        # tokenizer = "bert-base-uncased"
        datasets = get_datasets_fromdf(train_df, val_df, test_df, "comment_text")
        tokenizer = AutoTokenizer.from_pretrained(values.bert_model)

        sampling_agent = SamplingAgent(heldout_df.sample(n=200), train_df, device=device)
        model = TransformersTextClassification(
            values.bert_model, datasets["num_classes"]
        )

        sampling_agent.sample(
            100, "cvirs", model=model, values=values, tokenizer=values.bert_model
        )
        #print(sampling_agent.subword_dict)
        #dict_values = list(sampling_agent.subword_dict.values())
        #print(np.average([x["mean"] for x in dict_values]))
        # train_wrapper(model, train_df, val_df, test_df, values, ep=1)
        # sampling_agent.finetune_bert_lm(AutoTokenizer.from_pretrained("distilbert-base-uncased"))
        # print(sampling_agent.data_name)
        # sampling_agent.sample(100, "masked_lm", model=model, values=values)
