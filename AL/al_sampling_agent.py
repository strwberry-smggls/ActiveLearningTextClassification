import pandas as pd
import pytorch_lightning as pl
import torchmetrics
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from torch.nn.modules import loss
from torch import nn
import numpy
from nltk.tokenize import word_tokenize
import nltk
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    AutoTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    AutoModelForMaskedLM,
)
import pickle
import random
from tqdm.auto import tqdm
from transformers import AdamW
import time
import math
from transformers_multilabel_cap import *
import fasttext
 
class BookDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        input_ids = self.encodings["input_ids"][index]
        labels = self.encodings["labels"][index]
        attention_mask = self.encodings["attention_mask"][index]
        token_type_ids = self.encodings["token_type_ids"][index]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


class SamplingAgent:
    def __init__(
        self, heldout_df, train_df, model=None, data_name=None, seed=12, device="cpu"
    ):

        self.heldout_data = heldout_df
        self.training_data = train_df
        self.model = model
        self.subword_dict = None
        self.mlm_dict = None
        self.data_name = data_name
        self.seed = seed
        self.device = device
        self.alps_model = None
        self.input_column = "comment_text"
        # self.input_column = text_column
        self.label_column = "labels"
        self.dal_model_name = "distilbert-base-uncased"
        self.dal_model = None

        self.call_counter = 0
        # self.dal_coll = transformers_collator.TransformersCollator(
        #    self.dal_model_name)

    def sample(self, n_sample, strategy, model=None, tokenizer=None, values=None):
        print(f"SAMPLING AGENT: Sampling {n_sample} sentences with strategy {strategy}")
        print(strategy)
        if strategy == "random" or strategy == "finetune":
            sampled = self.heldout_data.sample(n=n_sample, random_state=self.seed)
        elif strategy == "combine":
            dal_uncertainties = sorted(self.dal(values), key=lambda x:x[0])
            alps_indices = self.alps(tokenizer, len(self.heldout_data))
            if self.subword_dict == None:
                self.subword_count(values)
            subword_scores = sorted([(x[0], x[1]["mean"]) for x in list(self.subword_dict.items())], key=lambda x:x[0])
            alps_scores = sorted([(alps_indices[i],1/(i+1)) for i in range(0,len(alps_indices))],key=lambda x:x[0])
            print(len(subword_scores))
            print(len(dal_uncertainties))
            print(len(alps_indices))
            print(len(self.heldout_data))
            #input("...")
            combined_scores = []
            for i in range(len(alps_scores)):
                alp = alps_scores[i]
                dal = dal_uncertainties[i]
                sw = subword_scores[i]
                cs = (alp[1] + dal[1] + sw[1])/3
                combined_scores.append((alp[0],cs))
            combined_scores = sorted(combined_scores, key=lambda x:x[1], reverse=True)[:n_sample]
            sampled = self.heldout_data.loc[[i[0] for i in combined_scores],:]

        elif strategy == "entropy":
            entropies = self.entropy_ranking(model, tokenizer)[:n_sample]
            sampled = self.heldout_data.loc[[i[0] for i in entropies], :]
        elif strategy == "cvirs":
            if values.model_type == "fasttext":
                model = fasttext.load_model(f"{values.checkpoint_path}_fasttext.bin")
                uncertainties = sorted(self.margin_ranking_fasttext(model, values), reverse=True, key=lambda x:x[1])[:n_sample]
            else:
                uncertainties = self.margin_ranking(model, values)[:n_sample]
            
            #sampled = self.heldout_data.loc[[i[0] for i in uncertainties], :]
            sampled = self.heldout_data.loc[self.heldout_data.iloc[:,0].isin([x[0] for x in uncertainties])]
            
        elif strategy == "dal":
            uncertainties = self.dal(values)[:n_sample]
            sampled = self.heldout_data.loc[[i[0] for i in uncertainties], :]
        elif strategy == "core":
            sampled = self.core_set(n_sample)
        elif strategy == "alps":
            indices = self.alps(tokenizer, n_sample)
            sampled = self.heldout_data.loc[[i for i in indices], :]
        elif strategy == "subword":
            """
            if not self.subword_dict:
                # try and load pre-computed subword ranking
                try:
                    if self.data_name == None:
                        self.subword_dict = pickle.load(open("subword_dict.pkl", "rb"))
                    else:
                        self.subword_dict = pickle.load(
                            open(f"subword_dict_{self.data_name}.pkl", "rb")
                        )

                except:
                    self.subword_count(values)
            """
            if self.subword_dict == None:
                self.subword_count(values)
            # print(self.subword_dict.items())
            by_count = sorted(
                self.subword_dict.items(), reverse=True, key=lambda x: x[1]["mean"]
            )[:n_sample]
            for s in by_count:
                self.subword_dict.pop(s[0], None)

            #print(self.heldout_data.head())
            #print(by_count)
            #sampled = self.heldout_data.loc[[i[0] for i in by_count], :]
            #print(by_count)
            sampled = self.heldout_data.loc[self.heldout_data.iloc[:,0].isin([x[0] for x in by_count])]
            #print(sampled)
        elif strategy == "masked_lm":
            """
            if self.mlm_dict == None:
                # try and load pre-computed mlm ranking
                try:
                    if self.data_name == None:
                        self.mlm_dict = pickle.load(open("mlm_dict.pkl", "rb"))
                    else:
                        self.mlm_dict = pickle.load(
                            open(f"mlm_dict_{self.data_name}.pkl", "rb")
                        )
                except:
            """
            if self.mlm_dict == None:
                self.masked_lm()

            by_count = sorted(self.mlm_dict.items(), key=lambda x: x[1]["acc"])[
                :n_sample
            ]
            print(by_count)

            # print(f"selected {by_count}")
            # input("...")q
            for s in by_count:
                self.mlm_dict.pop(s[0])
            indices = [i[0] for i in by_count]
            indices_new = []
            for i in indices:
                try:
                    row = self.heldout_data.loc[i, :]
                    indices_new.append(i)
                except:
                    # indices_new.append(random.randint(0, len(self.heldout_data)))
                    indices_new.append(random.choice(self.heldout_data.index.tolist()))
            sampled = self.heldout_data.loc[[i for i in indices_new], :]

        self.drop_from_heldout(sampled)
        self.training_data = pd.concat([self.training_data, sampled])
        print(
            f"SAMPLING AGENT: heldout data size: {len(self.heldout_data)}, train data size: {len(self.training_data)}"
        )
        return self.training_data

    def safe_divide(self, a, b):
        try:
            ret = a / b
        except ZeroDivisionError:
            ret = 0
        return ret

    def entropy_ranking(self, model, model_name):
        print("collecting model entropy on heldout data...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        # model.freeze()

        sent_entropy = []
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(
            self.heldout_data["comment_text"].tolist(),
            truncation=True,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
        ).to(device)
        batch_size = 4
        entropies = []
        inputs_list = []
        inputs_list.extend(inputs["input_ids"].cpu().numpy())
        for pred_batch in tqdm(range(math.ceil(len(inputs["input_ids"]) / batch_size))):
            batch_start = pred_batch * batch_size
            batch_end = pred_batch * batch_size + batch_size
            # print(batch_start)
            # print(batch_end)
            if batch_end > len(inputs["input_ids"]):
                batch_end = len(inputs["input_ids"])
            encoded_batch = inputs["input_ids"][batch_start:batch_end]
            attention_masks = inputs["attention_mask"][batch_start:batch_end]
            # ttids = inputs["token_type_ids"][batch_start:batch_end]

            logit = model(encoded_batch, attention_masks).detach()
            ent_list = []
            for l in logit:
                l = torch.softmax(l, dim=-1).cpu()
                for prob in l:
                    ent_list.append(prob * (1 / numpy.log(prob)))
            entropy = sum(ent_list)
            entropies.append(entropy)

        sent_entropy = [
            (self.heldout_data.index[i], entropies[i]) for i in range(len(entropies))
        ]
        # print(sent_entropy)
        return sorted(sent_entropy, reverse=True, key=lambda x: x[1])

    def cat_entropy(self, v, w):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        v_ones = len([1 for x in v if x == 1])
        v_zeros = len(v) - v_ones
        w_ones = len([1 for x in w if x == 1])
        w_zeros = len(v) - w_ones

        for i in range(len(v)):
            v_ = v[i]
            w_ = w[i]
            if v_ == 1:
                if w_ == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if w_ == 1:
                    fn += 1
                else:
                    tn += 1

        return self.safe_divide(
            2 * self.entropy4(v_count=2, tp=tp, fp=fp, fn=fn, tn=tn, num_labels=len(v))
            - self.entropy4(v_count=1, zeros=v_zeros, ones=v_ones)
            - self.entropy4(v_count=1, zeros=w_zeros, ones=w_ones),
            self.entropy4(v_count=2, tp=tp, fp=fp, fn=fn, tn=tn, num_labels=len(v)),
        )

    def entropy4(self, v_count, tp=0, fp=0, fn=0, tn=0, ones=0, zeros=0, num_labels=0):
        if v_count == 1:
            return self.entropy2(ones, zeros)
        elif v_count == 2:
            return self.entropy2((fp + fn) / num_labels, (tp + tn) / num_labels) + (
                fp + fn
            ) / num_labels * self.entropy2(
                self.safe_divide(fp, (fp + fn)),
                self.safe_divide(fn, (fp + fn))
                + (tp + tn)
                / num_labels
                * (
                    self.entropy2(
                        self.safe_divide(tp, (tp + tn)), self.safe_divide(tn, (tp + tn))
                    )
                ),
            )

        pass

    def entropy2(self, a, b):
        if a <= 0:
            t1 = 0
        else:
            t1 = -a * math.log2(a)

        if b <= 0:
            t2 = 0
        else:
            t2 = -b * math.log2(b)
        return t1 - t2

    def hamming_distance(self, v, w):
        if len(v) != len(w):
            print(
                f"can't compute hamming distance, vectors are of different length: {len(v)}, {len(w)}"
            )
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(v)):
            v_ = v[i]
            w_ = w[i]
            if v_ == 1:
                if w_ == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if w_ == 1:
                    fn += 1
                else:
                    tn += 1

        return (fp + fn) / len(v)

    def margin_ranking_fasttext(self, model, values):
        # torch.set_grad_enabled(False)
        all_margins = []

        print("starting classification margin computation")
        print("")
        count = 0
        #tok = AutoTokenizer.from_pretrained(values.bert_model)
        #inputs = tok(
        #    self.heldout_data["comment_text"].tolist(),
        #    truncation=True,
        #    return_tensors="pt",
        #    max_length=512,
        #    padding="max_length",
        #).to(device)
        #model.eval()
        #model.to(device)
        # labels = inputs["input_ids"]
        # logits = []
        # inputs_list = []
        # inputs_list.extend(inputs["input_ids"].cpu().numpy())
        batch_size = 1
        margin_list = []
        i = 0
        index_to_margin = []
        index_to_catent = {}
        # t1 = time.time()
        texts = []
        for index, row in self.heldout_data.iterrows():
            #logit = model.predict(row["comment_text"], k=-1)[1]
            texts.append(row["comment_text"].replace("\n", " "))
        logits = model.predict(texts, k=-1)[1]
        
        for logit in tqdm(logits):
            # calculating classification margin for each entry
            l = logit
            l_sig = logit
            l = torch.softmax(torch.tensor(l), dim=-1)
            margin = [abs(x - (1 - x)) for x in l]
            index_to_margin.append((self.heldout_data.index[i], margin))

            # thresholding predictions for category vector calculation
            threshold = 0.1
            l_pred = []
            for entry in l_sig:
                if entry >= threshold:
                    l_pred.append(0)
                else:
                    l_pred.append(1)

            cat_ent = 0
            for index, row in self.training_data.iterrows():
                labels = row[
                        self.training_data.columns.get_loc("comment_text") + 1 :
                    ].to_list()
                #print(l_pred)
                #print(labels)
                #input("...")
                hamming = self.hamming_distance(l_pred, labels)
                if hamming < 1:
                    cat_ent += self.cat_entropy(l_pred, labels)
                else:
                    cat_ent += 1
            cat_ent = cat_ent / len(self.training_data)
            index_to_catent[self.heldout_data.index[i]] = cat_ent

            i += 1
        
        """
        for index, row in self.heldout_data.iterrows():
            # if index > 100:
            #    break
            text = row["input"]
            labels = [int(l) for l in row["labels"].split(" ") if l]

            encoded = self.coll([{"input": text, "labels": labels}])

            probs = torch.softmax(self.model(encoded))[0].numpy()

            margins = [abs(x - (1 - x)) for x in probs]
            all_margins.append((index, margins))
            count += 1
            print(f"{count/len(self.heldout_data)*100:.2f}%", end="\r")
        """
        label_rankings = {}
        for label_index in range(len(index_to_margin[0][1])):
            sorted_margins = sorted(index_to_margin, key=lambda x: x[1][label_index])
            label_rankings[label_index] = [x[0] for x in sorted_margins]

        uncertainties = []
        print(f"getting category vectors...")
        for i, row in tqdm(self.heldout_data.iterrows()):
            # if i > 100:
            #    break
            unc = 0
            for l in label_rankings:
                ranking = label_rankings[l]
                unc += len(self.heldout_data) - ranking.index(i)
            unc = unc / (len(label_rankings) * (len(all_margins) - 1))
            unc = unc * index_to_catent[i]
            uncertainties.append((i, unc))

        # t2 = time.time()
        # print(t2 - t1)
        return uncertainties
    def margin_ranking(self, model, values):
        # torch.set_grad_enabled(False)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        all_margins = []

        print("starting classification margin computation")
        print("")
        count = 0
        tok = AutoTokenizer.from_pretrained(values.bert_model)
        inputs = tok(
            self.heldout_data["comment_text"].tolist(),
            truncation=True,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
        ).to(device)
        model.eval()
        model.to(device)
        # labels = inputs["input_ids"]
        # logits = []
        # inputs_list = []
        # inputs_list.extend(inputs["input_ids"].cpu().numpy())
        batch_size = 16
        margin_list = []
        i = 0
        index_to_margin = []
        index_to_catent = {}
        # t1 = time.time()
        for pred_batch in tqdm(range(math.ceil(len(inputs["input_ids"]) / batch_size))):
            batch_start = pred_batch * batch_size
            batch_end = pred_batch * batch_size + batch_size
            # print(batch_start)
            # print(batch_end)
            if batch_end > len(inputs["input_ids"]):
                batch_end = len(inputs["input_ids"])
            encoded_batch = inputs["input_ids"][batch_start:batch_end]
            attention_masks = inputs["attention_mask"][batch_start:batch_end]
            logit = model(encoded_batch, attention_masks).detach()
            for l in logit:
                # calculating classification margin for each entry
                l_sig = torch.sigmoid(l).cpu().numpy()
                l = torch.softmax(l, dim=-1).cpu().numpy()
                margin = [abs(x - (1 - x)) for x in l]
                index_to_margin.append((self.heldout_data.iloc[i, 0], margin))

                # thresholding predictions for category vector calculation
                threshold = 0.5
                l_pred = []
                for entry in l_sig:
                    if entry >= threshold:
                        l_pred.append(0)
                    else:
                        l_pred.append(1)

                cat_ent = 0
                for index, row in self.training_data.iterrows():
                    labels = row[
                        self.training_data.columns.get_loc("comment_text") + 1 :
                    ].to_list()
                    hamming = self.hamming_distance(l_pred, labels)
                    if hamming < 1:
                        cat_ent += self.cat_entropy(l_pred, labels)
                    else:
                        cat_ent += 1
                cat_ent = cat_ent / len(self.training_data)
                index_to_catent[self.heldout_data.iloc[i, 0]] = cat_ent

                i += 1
        """
        for index, row in self.heldout_data.iterrows():
            # if index > 100:
            #    break
            text = row["input"]
            labels = [int(l) for l in row["labels"].split(" ") if l]

            encoded = self.coll([{"input": text, "labels": labels}])

            probs = torch.softmax(self.model(encoded))[0].numpy()

            margins = [abs(x - (1 - x)) for x in probs]
            all_margins.append((index, margins))
            count += 1
            print(f"{count/len(self.heldout_data)*100:.2f}%", end="\r")
        """
        label_rankings = {}
        for label_index in range(len(index_to_margin[0][1])):
            sorted_margins = sorted(index_to_margin, key=lambda x: x[1][label_index])
            label_rankings[label_index] = [x[0] for x in sorted_margins]

        uncertainties = []
        print(f"getting category vectors...")
        for i, row in tqdm(self.heldout_data.iterrows()):
            # if i > 100:
            #    break
            unc = 0
            for l in label_rankings:
                ranking = label_rankings[l]
                unc += len(self.heldout_data) - ranking.index(row.iloc[0])
            unc = unc / (len(label_rankings) * (len(all_margins) - 1))
            unc = unc * index_to_catent[row.iloc[0]]
            uncertainties.append((row.iloc[0], unc))

        # t2 = time.time()
        # print(t2 - t1)
        return sorted(uncertainties, reverse=True, key=lambda x: x[1])

    def drop_from_heldout(self, sampled: pd.DataFrame):
        self.heldout_data = pd.concat([self.heldout_data, sampled]).drop_duplicates(
            keep=False
        )
        # self.heldout_data = self.heldout_data.reset_index(drop=True)

    def dal_train_model(self, values):
        dal_epochs = 2
        tokenizer = AutoTokenizer.from_pretrained(values.bert_model)
        self.dal_model = FFTextClassification(
            num_classes=2, input_size=tokenizer.vocab_size
        )
        self.dal_model.to(device)
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
        sents = self.training_data[self.input_column].to_list()
        sents.extend(self.heldout_data[self.input_column].to_list())
        discr_labels = [[0, 1] for i in range(len(self.training_data))]
        discr_labels.extend([[1, 0] for i in range(len(self.heldout_data))])

        dal_train = pd.DataFrame(
            {
                self.input_column: sents,
                "l1": [x[0] for x in discr_labels],
                "l2": [x[1] for x in discr_labels],
            }
        )
        dal_train, dal_val = train_test_split(dal_train, test_size=10, random_state=101)
        datasets = get_datasets_fromdf(
            dal_train, dal_val, pd.DataFrame({}), "comment_text"
        )
        checkpoint_path = values.checkpoint_path + f"dal-model_checkpoint.pt"
        best_model_path = values.checkpoint_path + f"dal-model__best-model.pt"

        # compute training and warmup steps needed for the scheduler
        num_training_steps = len(datasets["train"]) // values.batch_size

        num_warmup_steps = 0.06 * num_training_steps
        # get the optimizer and scheduler
        opt_sch = get_optimizer(
            model=self.dal_model,
            lr=values.learning_rate,
            epochs=dal_epochs,
            tr_steps=num_training_steps,
            wrm_steps=num_warmup_steps,
        )
        optimizer = opt_sch["optimizer"]
        scheduler = opt_sch["scheduler"]
        # load best model
        # checkpoint_path = values.checkpoint_path
        # best_model_path = values.best_model_path
        validation_loader = DataLoader(
            datasets["val"], collate_fn=BertCollator(values.bert_model), **test_params
        )
        training_loader = DataLoader(
            datasets["train"],
            collate_fn=BertCollator(values.bert_model),
            **train_params,
        )
        print(f"Training with {len(dal_train)} sentences")
        train_model(
            n_epochs=dal_epochs,
            training_loader=training_loader,
            validation_loader=validation_loader,
            model=self.dal_model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=checkpoint_path,
            best_model_path=best_model_path,
            force_epochs=True,
        )

    def sentence_dal(self, sentence):
        if self.dal_model == None:
            self.dal_train_model()

        input = self.dal_coll([{"input": sentence, "labels": 0}])
        logits = self.dal_model(input)
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(logits)[0].detach().numpy()
        certainty = max(probs) - min(probs)
        return certainty

    def dal(self, values):
        # del (self.model)
        t1 = time.time()
        self.dal_train_model(values)
        t2 = time.time()
        print(t2 - t1)
        batch_size = 12
        tokenizer = AutoTokenizer.from_pretrained(self.dal_model_name)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # calculate surprise embeddings for every sent in heldout data
        print("calculating DAL uncertainty")
        inputs = tokenizer(
            self.heldout_data["comment_text"].tolist(),
            truncation=True,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
        ).to(device)
        self.dal_model.train()
        # labels = inputs["input_ids"]
        # logits = []
        inputs_list = []
        inputs_list.extend(inputs["input_ids"].cpu().numpy())
        margin_list = []
        for pred_batch in tqdm(range(math.ceil(len(inputs["input_ids"]) / batch_size))):
            batch_start = pred_batch * batch_size
            batch_end = pred_batch * batch_size + batch_size
            # print(batch_start)
            # print(batch_end)
            if batch_end > len(inputs["input_ids"]):
                batch_end = len(inputs["input_ids"])
            encoded_batch = inputs["input_ids"][batch_start:batch_end]
            attention_masks = inputs["attention_mask"][batch_start:batch_end]
            logit = self.dal_model(encoded_batch, attention_masks).detach()

            for l in logit:
                l = torch.softmax(l, dim=-1).cpu().numpy()
                # margin = l[numpy.argmax(l)] - l[numpy.argmin(l)]
                margin = l[1]
                margin_list.append(margin)

            # print(margin_list)

        sent_margins = [
            (self.heldout_data.index[i], margin_list[i])
            for i in range(len(margin_list))
        ]
        # print(sent_margins)
        # print(len(self.heldout_data))
        # print(len(margin_list))

        return sorted(sent_margins, key=lambda x: x[1], reverse=True)

    def core_set(self, n_sample):

        # greedy k-center
        sampled = []
        while len(sampled) < n_sample:
            # find argmax_i min_j dist(x_i, x_j)
            minimums = []
            for index, x_i in self.heldout_data.iterrows():
                distances = []
                for index, x_j in self.training_data.iterrows():
                    distances.append(self.l2_activation_distance(x_i, x_j))
                minimums.append(min(distances))
            max_index = minimums.index(max(minimums))
            sampled.append(self.heldout_data.iloc[max_index])
            self.heldout_data.drop(max_index)
        return sampled

    def subword_count(self, values):
        # ratio of subwords to original word count per sample
        # higher ratio means more rare words the LM didn't know
        self.subword_dict = {}
        nltk.download("punkt")
        tokenizer = AutoTokenizer.from_pretrained(values.bert_model)
        # t1 = time.time()
        for index, t in tqdm(self.heldout_data.iterrows()):
            # if index > 1000:
            #    break
            nltk_tok = word_tokenize(t["comment_text"])
            subword_length = 0
            for word in nltk_tok:
                x_encoded = tokenizer(
                    word, return_tensors="pt", add_special_tokens=False
                )["input_ids"][0]
                # x_decoded = self.coll.tokenizer.decode(x_encoded)
                """
                teststr = "arteriosclerosis"
                test_enc = self.coll.tokenizer(teststr,
                                               add_special_tokens=False)
                print(len(test_enc["input_ids"]))
                print(self.coll.tokenizer.decode(test_enc["input_ids"]))
                """
                subword_length += len(x_encoded)
            stat_dict = {"mean": subword_length / len(nltk_tok)}
            
            self.subword_dict[t.iloc[0]] = stat_dict
            #self.subword_dict[index] = stat_dict
        # t2 = time.time()
        # print(t2 - t1)
        if self.data_name == None:
            pickle.dump(self.subword_dict, open("subword_dict.pkl", "wb"))
        else:
            pickle.dump(
                self.subword_dict,
                open(f"subword_dict_{self.data_name}.pkl", "wb"),
            )

    def masked_lm(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        model.eval()
        model.to(0)

        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        self.mlm_dict = {}
        print(f"calculating masked_lm scores...")
        for index, t in tqdm(self.heldout_data.iterrows()):
            # if index > 100:
            #    break
            word_tok = tokenizer.tokenize(t["comment_text"])[:510]
            pos = nltk.pos_tag(word_tok)
            word_tok.insert(0, "[CLS]")
            word_tok.append("[SEP]")

            correct = 0
            total = 0
            i = 0
            # print(f"sentence is {word_tok}")
            for word_pos in pos:
                input_tensors = []
                label_tensors = []
                if word_pos[1].startswith(("N", "V", "R", "J")):
                    # print(f"found {word_pos[0]} at {word_pos[1]}")
                    # print(f"setting {word_tok[i]} to [MASK]")

                    label_tokens = tokenizer.convert_tokens_to_ids(word_tok)
                    indexed_labels = torch.tensor([label_tokens]).to(0)
                    word_tok_masked = [x for x in word_tok]
                    word_tok_masked[i] = "[MASK]"
                    indexed_tokens = tokenizer.convert_tokens_to_ids(word_tok_masked)[
                        :512
                    ]
                    tokens_tensor = torch.tensor([indexed_tokens]).to(0)
                    input_tensors.append(tokens_tensor)
                    label_tensors.append(indexed_labels)
                    # print(
                    #    f"giving {word_tok_masked[:100]}... with length {len(word_tok_masked)} to bert tokenizer"
                    # )

            with torch.no_grad():
                if len(input_tensors) > 0:
                    outputs = model(
                        torch.stack(input_tensors), labels=torch.stack(label_tensors)
                    )
                    predictions = outputs.logits
                    loss = outputs.loss
                    print(outputs)
                    print(outputs.shape)
                    input("...")
                    pred_index = torch.argmax(predictions[0, i]).item()
                    predicted_token = tokenizer.convert_ids_to_tokens([pred_index])[0]

                    if predicted_token == word_tok[i]:
                        correct += 1
                    total += 1

            i += 1
            # if there was no mlm prediction, disregard the sentence (i.e. max rank)
            if total == 0:
                acc = 1
            else:
                acc = correct / total
            self.mlm_dict[index] = {"acc": acc}

            if self.data_name == None:
                pickle.dump(self.mlm_dict, open("mlm_dict.pkl", "wb"))
            else:
                pickle.dump(self.mlm_dict, open(f"mlm_dict_{self.data_name}.pkl", "wb"))
            # print(f"{index/len(self.heldout_data)*100}%", end="\r")

    def l2_activation_distance(self, x, y):
        x_encoded = self.coll.tokenizer(
            x["input"],
            add_special_tokens=True,
            padding="longest",
            max_length=self.coll.max_length,
            truncation=True,
            return_attention_mask=True,
            return_length=True,
            return_tensors="pt",
        )
        y_encoded = self.coll.tokenizer(
            y["input"],
            add_special_tokens=True,
            padding="longest",
            max_length=self.coll.max_length,
            truncation=True,
            return_attention_mask=True,
            return_length=True,
            return_tensors="pt",
        )
        x_out = self.model(x_encoded)[0].detach().numpy()
        y_out = self.model(y_encoded)[0].detach().numpy()

        l2 = numpy.linalg.norm(x_out - y_out)
        return l2

    def finetune_bert_lm(self, tokenizer):
        train_data = self.heldout_data["comment_text"].tolist()

        inputs = tokenizer(
            train_data,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].detach().clone()
        random_tensor = torch.rand(inputs["input_ids"].shape)
        masked_tensor = (
            (random_tensor < 0.15)
            * (inputs["input_ids"] != 101)
            * (inputs["input_ids"] != 102)
            * (inputs["input_ids"] != 0)
        )
        # getting all those indices from each row which are set to True, i.e. masked.
        nonzeros_indices = []
        for i in range(len(masked_tensor)):
            nonzeros_indices.append(torch.flatten(masked_tensor[i].nonzero()).tolist())
        # setting the values at those indices to be a MASK token (103) for every row in the original input_ids.
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i, nonzeros_indices[i]] = 103
        dataset = BookDataset(inputs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        epochs = 2
        optimizer = AdamW(self.alps_model.parameters(), lr=1e-5)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.alps_model.train()
        self.alps_model.to(device)
        for epoch in range(epochs):
            loop = tqdm(dataloader)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = self.alps_model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                loop.set_description("Epoch: {}".format(epoch))
                loop.set_postfix(loss=loss.item())
        self.alps_model.save_pretrained(f"helper_models/alps_model")

    def alps(self, tokenizer_name, n_sample=100):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # calculate surprise embeddings for every sent in heldout data
        self.alps_model = AutoModelForMaskedLM.from_pretrained(tokenizer_name).to(
            device
        )
        print("calculating surprise embeddings")
        inputs = tokenizer(
            self.heldout_data["comment_text"].tolist(),
            truncation=True,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
        ).to(device)
        labels = inputs["input_ids"].to(device)
        batch_size = 12
        logits = []
        # inputs_list = []
        # inputs_list.extend(inputs["input_ids"].cpu().numpy())
        embeddings = []
        running_i = 0
        indices = []
        for pred_batch in tqdm(range(math.ceil(len(inputs["input_ids"]) / batch_size))):
            batch_start = pred_batch * batch_size
            batch_end = pred_batch * batch_size + batch_size
            # print(batch_start)
            # print(batch_end)
            if batch_end > len(inputs["input_ids"]):
                batch_end = len(inputs["input_ids"])
            encoded_batch = inputs["input_ids"][batch_start:batch_end]
            attention_masks = inputs["attention_mask"][batch_start:batch_end]
            ttids = inputs["token_type_ids"][batch_start:batch_end]
            encoded_labels = labels[batch_start:batch_end]
            logit = self.alps_model(
                encoded_batch,
                labels=encoded_labels,
                attention_mask=attention_masks,
                token_type_ids=ttids,
            ).logits.detach()

            for position in range(len(encoded_batch)):
                encoded_input = encoded_batch[position].detach()
                l = logit[position]
                probs = torch.softmax(l, dim=-1)

                output_seq = []
                output_probs = []
                for p in probs.squeeze():
                    p = p.cpu().numpy()
                    max_ind = numpy.argmax(p)
                    output_seq.append(max_ind)
                    output_probs.append(p[max_ind])
                sampled_indices = numpy.random.choice(
                    range(len(output_seq)), size=int(len(output_seq) / 100 * 15)
                )
                # print(sampled_indices)
                # input("...")
                sp_embedding = [0 for i in range(len(output_seq))]
                # print(output_seq)
                # print(tokenizer.decode(output_seq))
                # print(encoded_input)
                for i in sampled_indices:
                    pred_tok = output_seq[i]
                    gold_tok = encoded_input[i]
                    pred_prob = output_probs[i]

                    # print(pred_tok)
                    # print(gold_tok)
                    # input("...")

                    if pred_tok == gold_tok:
                        if pred_prob == 0:
                            sp_embedding[i] = 0
                        else:
                            sp_embedding[i] = -numpy.log(pred_prob)
                    else:
                        ce = 1 - pred_prob
                        if ce == 0:
                            sp_embedding[i] = 0
                        else:
                            sp_embedding[i] = -numpy.log(1 - pred_prob)

                embeddings.append(sp_embedding)
                # print(embeddings)
                indices.append(self.heldout_data.index[running_i])
                running_i += 1

        del inputs
        """
        embeddings = []
        indices = []

        running_i = 0

        for index, row in tqdm(self.heldout_data.iterrows()):
            # t1 = time.time()
            encoded_input = labels[running_i].cpu().numpy()
            l = logits[running_i].cpu()
            probs = torch.softmax(l, dim=-1)
            output_seq = []
            output_probs = []
            # t2 = time.time()
            # print(f"softmax takes {t2 - t1} seconds")
            for p in probs.squeeze():
                p = p.numpy()
                max_ind = numpy.argmax(p)
                output_seq.append(max_ind)
                output_probs.append(p[max_ind])

            # print(output_seq)
            # print(output_probs)
            # print(len(output_seq))
            # print(len(output_probs))

            # t1 = time.time()
            sampled_indices = numpy.random.choice(
                range(len(output_seq)), size=int(len(output_seq) / 100 * 15)
            )
            # print(sampled_indices)
            # input("...")
            sp_embedding = [0 for i in range(len(output_seq))]
            # print(output_seq)
            # print(tokenizer.decode(output_seq))
            # print(encoded_input)
            for i in sampled_indices:
                pred_tok = output_seq[i]
                gold_tok = encoded_input[i]
                pred_prob = output_probs[i]

                # print(pred_tok)
                # print(gold_tok)
                # input("...")

                if pred_tok == gold_tok:
                    sp_embedding[i] = -numpy.log(pred_prob)
                else:
                    sp_embedding[i] = -numpy.log(1 - pred_prob)
            embeddings.append(sp_embedding)
            indices.append(index)
            running_i += 1
            # t2 = time.time()
            # print(f"collecting surprisal embeddings takes {t2 - t1} seconds")
        # print(embeddings)
        # print(indices)
        # print(len(embeddings))
        # print(len(indices))
        # input("...")
        """
        embeddings = normalize(embeddings)
        print("clustering surprise embeddings")
        clustering = KMeans(n_clusters=n_sample, random_state=0).fit(embeddings)
        distances = {}
        print(f"getting closest sentences to {n_sample} cluster centers")
        for i in tqdm(range(len(embeddings))):
            sent = embeddings[i]
            sent_index = indices[i]
            clust_label = clustering.labels_[i]
            clust_center = clustering.cluster_centers_[clust_label]
            if clust_label not in distances:
                distances[clust_label] = []
            dist = cosine_distances([sent], [clust_center])
            distances[clust_label].append((sent_index, dist))
        final_indices = []
        for d in distances:
            final_indices.append(sorted(distances[d], key=lambda x: x[1])[0][0])
        return final_indices

    def softmax(self, t):
        exponentials = numpy.exp(t)
        sfmax = [x / sum(exponentials) for x in exponentials]
        return sfmax

    def update(self, heldout, train, coll=None, model=None):
        self.heldout_data = heldout
        self.training_data = train
        if coll != None:
            self.coll = coll
        if model != None:
            self.model = model


"""
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mlm":
"""
