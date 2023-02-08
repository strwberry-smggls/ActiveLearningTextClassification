from typing import Dict, List, Callable, Union
import hydra
from omegaconf import DictConfig
import pandas as pd
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn import metrics
from torch import nn

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import os

# torch.manual_seed(9)

if torch.cuda.is_available():
    device = torch.device("cuda")

    print("There are %d GPU(s) available." % torch.cuda.device_count())

    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def loss_fn(outputs: List, targets: List):
    """BCE loss function with logits."""
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def get_optimizer(
    model: nn.Module, lr: float, epochs: int, tr_steps: int, wrm_steps: int
):
    """Configure all optimizers and schedulers.

    Args:
        model (nn.Module): The DL model.
        lr (float): Learning rate.
        epochs (int): Number of epochs used for training the model.
        tr_steps (int): Number of training steps.
        wrm_steps (int): Number of warmup steps.
    Returns:
        Dict: Dictionary containing the optimizer and the scheduler.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=wrm_steps,
        num_training_steps=epochs * tr_steps,
    )

    return {"optimizer": optimizer, "scheduler": scheduler}


class TransformersTextClassification(nn.Module):
    """Class for transformers models for text classification with linear
    pooling."""

    def __init__(self, pretrained_name_or_path: str, num_classes: int, seed:int = 0):
        """Constructor for the linear transformers classification model.

        Args:
            pretrained_name_or_path (str): Name of the pretrained
            transformers model or path to the pretrained model.
            num_classes (int): Number of classes.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.num_classes = num_classes
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_name_or_path,
            num_labels=self.num_classes,
        )

    def forward(self, ids, msk):
        """Forward pass for the transformer classification model that for an
        input returns the logit."""

        # Apply model model on the input x
        y = self.model(ids, attention_mask=msk).logits

        return y

    def get_pretrained_name_or_path(self) -> str:
        """Get the pretrained model name or the path to a pretrained model."""
        return self.model.name_or_path

    def get_model_name(self) -> str:
        """Get the name of the model (e.g. BertModel)."""
        return self.model._get_name()


class FFTextClassification(nn.Module):
    """Class for transformers models for text classification with linear
    pooling."""

    def __init__(self, num_classes: int, input_size: int):
        """Constructor for the linear transformers classification model.

        Args:
            pretrained_name_or_path (str): Name of the pretrained
            transformers model or path to the pretrained model.
            num_classes (int): Number of classes.
        """
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.num_classes = num_classes
        self.vocab_size = input_size
        # Model
        self.l1 = torch.nn.Linear(input_size, 256).to(self.device)
        self.l2 = torch.nn.Linear(256, 256).to(self.device)
        self.l3 = torch.nn.Linear(256, num_classes).to(self.device)
        self.relu = torch.nn.ReLU()
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #    pretrained_model_name_or_path=pretrained_name_or_path,
        #    num_labels=self.num_classes,
        # )

    def forward(self, ids, msk=None):
        """Forward pass for the transformer classification model that for an
        input returns the logit."""

        # Apply model model on the input x
        # print(ids)
        input_ids = [0 for i in range(self.vocab_size)]
        onehot_batch = []
        for id in ids:
            input_ids = [0 for i in range(self.vocab_size)]
            for d in id:
                input_ids[d] = 1
            onehot_batch.append(input_ids)
        onehot_batch = torch.tensor(onehot_batch).float().to(self.device)
        # input_ids = torch.tensor(input_ids).to(self.device)

        y = self.relu(self.l1(onehot_batch))
        y = self.relu(self.l2(y))
        y = self.l3(y)

        return y

    def get_pretrained_name_or_path(self) -> str:
        """Get the pretrained model name or the path to a pretrained model."""
        return self.model.name_or_path

    def get_model_name(self) -> str:
        """Get the name of the model (e.g. BertModel)."""
        return self.model._get_name()


class PretrainedTransformersTextClassification(nn.Module):
    def __init__(self, pretrained_name_or_path, num_classes, path=None):
        super(PretrainedTransformersTextClassification, self).__init__()
        # config = DistilBertConfig.from_pretrained('distilbert-base-uncased', output_hidden_states = True)
        if path:
            self.bert = AutoModel.from_pretrained(
                path,
                cache_dir="/home/users0/frommels/aih/cache",
                output_attentions=True,
            )
        else:
            self.bert = AutoModel.from_pretrained(
                pretrained_name_or_path,
                cache_dir="/home/users0/frommels/aih/cache",
                output_attentions=True,
            )
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # print(f"INPUT_IDS: {input_ids.shape}, ATTENTION_MASK: {attention_mask.shape}")

        pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
        ).pooler_output

        output = self.drop(pooled_output)
        output2 = self.out(output)
        # print(f"output size: {output2.size()}")
        return output2

    def get_pretrained_name_or_path(self) -> str:
        """Get the pretrained model name or the path to a pretrained model."""
        return self.model.name_or_path

    def get_model_name(self) -> str:
        """Get the name of the model (e.g. BertModel)."""
        return self.model._get_name()


class BertCollator:
    """Class for TransformersCollator."""

    def __init__(self, tokenizer_name):
        """Constructor for the BertCollator class."""
        super(BertCollator, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, batch):
        """Makes the collator object callable.

        Makes the collator object callable such that it can be passed to the
        collate_fn argument of any Pytorch DataLoader. This collator expects
        __getitem__ to return a dictionary with keys "text" and "labels". It
        will then tokenize the text and return the batch as a dictionary.

        Args:
            batch (List[Dict]): List of batch_size items.

        Returns:
            Dict: A dictionary containing the entire batch. The dictionary
                contains the following keys: text, input_ids, attention_mask
                and labels.
        """

        text, labels = zip(*[[sample["text"], sample["labels"]] for sample in batch])

        text = list(text)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "ids": encoding["input_ids"],
            "mask": encoding["attention_mask"],
            "labels": torch.FloatTensor(labels),
        }


class TextDataset(Dataset):
    """Class defining a TextDataset in Pytorch format."""

    def __init__(self, data: pd.DataFrame, text_column: str, label_columns: List[str]):
        """Constructor for the TextDataset.

        Args:
            data (pd.DataFrame): DataFrame containing the samples.
            text_column (str): Column containing the input text.
            label_columns (List[str]): Columns containing the labels.
        """
        super(TextDataset, self).__init__()

        self.data = data
        self.text_column = text_column
        self.label_columns = label_columns

    def __len__(self) -> int:
        """Returns the amount of samples in the Dataset."""

        return len(self.data)

    def __getitem__(self, index: int):
        """Returns the sample at position 'index'.

        Args:
            index (int): Position of the sample of interest in the Dataset.

        Returns:
            Dict: Dictionary containing the sample format as passed to the
                collator.
        """

        data_row = self.data.iloc[index]

        text = data_row[self.text_column]
        labels = data_row[self.label_columns]

        # convert from pd.Series to list
        labels = labels.values
        labels = labels.tolist()

        return {"text": text, "labels": labels}


def get_datasets(
    train_data_path: str, val_data_path: str, test_data_path: str, text_column: str
):
    """Generate train, val and test datasets from CSV files.
    Args:
        dataset_name(str): Name of the dataset.
        train_data_path (str): Path to the CSV file containing the training
            data.
        val_data_path (str): Path to the CSV file containing the validation
            data.
        test_data_path (str): Path to the CSV file containing the testing data.
        text_column (str): Name of the column containing the input text.
    Returns:
        Dict[str, MultilabelDataset, int]: A dictionary containing train,
        val and test MultilabelDatasets and the number of classes.
    """
    import pandas as pd

    train_df = pd.read_csv(train_data_path)
    train_df = train_df.sample(n=10000, random_state=24)
    columns = list(train_df.columns)
    columns = columns[columns.index(text_column) :]
    train_df = train_df[columns]
    val_df = pd.read_csv(val_data_path)
    val_df = val_df[columns]
    test_df = pd.read_csv(test_data_path)
    test_df = test_df[columns]

    train_dataset: Dataset = TextDataset(train_df, text_column, columns[2:])

    val_dataset: Dataset = TextDataset(val_df, text_column, columns[2:])

    test_dataset: Dataset = TextDataset(test_df, text_column, columns[2:])

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "num_classes": len(columns[2:]),
    }


def get_datasets_fromdf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
):
    """Generate train, val and test datasets from CSV files.
    Args:
        dataset_name(str): Name of the dataset.
        train_data_path (str): Path to the CSV file containing the training
            data.
        val_data_path (str): Path to the CSV file containing the validation
            data.
        test_data_path (str): Path to the CSV file containing the testing data.
        text_column (str): Name of the column containing the input text.
    Returns:
        Dict[str, MultilabelDataset, int]: A dictionary containing train,
        val and test MultilabelDatasets and the number of classes.
    """

    # train_df = pd.read_csv(train_data_path)
    columns = list(train_df.columns)
    columns = columns[columns.index(text_column) :]
    train_df = train_df[columns]

    # train_df = train_df.sample(n=10000, random_state=24)
    # val_df = pd.read_csv(val_data_path)
    val_df = val_df[columns]
    # test_df = pd.read_csv(test_data_path)

    train_dataset: Dataset = TextDataset(train_df, text_column, columns[1:])

    val_dataset: Dataset = TextDataset(val_df, text_column, columns[1:])
    if not test_df.empty:
        test_df = test_df[columns]
        test_dataset: Dataset = TextDataset(test_df, text_column, columns[1:])
    else:
        test_dataset = None

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "num_classes": len(columns[1:]),
    }


def load_ckp(checkpoint_fpath: str, model: nn.Module, optimizer: torch.optim) -> Dict:
    """Load a saved model.

    Args:
        checkpoint_fpath (str): Path to the saved model.
        model (nn.Module): Model to load checkpoint parameters into.
        optimizer (torch.optim): Optimizer used for model training.

    Returns:
          Dict: Dictionary of the loaded model along with the saved optimal
          probability threshold.
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint["optimizer"])
    # threshold
    thr = checkpoint["best_thr"]
    # return model, optimizer, epoch value, min validation loss
    return {"model": model, "thr": thr}


def save_ckp(
    state: Dict, is_best: bool, checkpoint_path: str, best_model_path: str
) -> None:
    """Function for saving a model.

    Args:
        state (Dict): Checkpoint with all infos we want to save.
        is_best (bool): Is it the best performing model (max F score).
        checkpoint_path (str): Path to save last model.
        best_model_path (str): Path to save best model.
    """
    f_path = checkpoint_path

    # path_only = "/".join(f_path.split("/")[:-1])+"/"
    # if not os.path.isdir(path_only):
    #    os.mkdir(path_only)

    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        print(f"saving best model to {best_fpath}")
        shutil.copyfile(f_path, best_fpath)


def find_best_cut(preds, targets) -> Dict:
    """Function that finds the best threshold with respect to the macro
    F score.

    Args:
        preds: Predictions probabilities.
        targets: Corresponding true labels.

    Returns:
        A dictionary of the optimal threshold and the best macro F score.
    """
    f1_max = -1
    final_thr = 0.5
    # the grid from which thresholds are selected
    thrl = [0.00001, 0.0001, 0.001, 0.01] + list(np.arange(0.05, 1, 0.05))  # + \
    # [0.9, 0.93, 0.97, 0.98]
    for thr in thrl:
        pred_thr = (np.array(preds) > thr).astype(int)
        f1_thr = metrics.f1_score(targets, pred_thr, average="macro")

        if f1_thr > f1_max:
            final_thr = thr
            f1_max = f1_thr

    return {"best_thr": final_thr, "best_f": f1_max}


def val_model(model: nn.Module, val_loader: DataLoader, eval_f1: bool = True):
    model.eval()
    val_targets = []
    val_outputs = []
    valid_loss = 0
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.float)
            outputs = model(ids, mask)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + loss.item()
            val_targets.extend(targets.detach().cpu().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy().tolist())

        # calculate validation Fscore

    valid_loss = valid_loss / len(val_loader)
    if eval_f1:
        best_cut = find_best_cut(val_outputs, val_targets)
        best_thr = best_cut["best_thr"]
        val_fscore = best_cut["best_f"]
    else:
        val_fscore = 1 - valid_loss
        best_thr = -1
    print(
        f"running validation only: average validation loss {valid_loss}; val F1 {val_fscore}"
    )
    return (val_fscore, valid_loss)


def train_model(
    n_epochs: int,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    scheduler: Callable,
    checkpoint_path: str,
    best_model_path: str,
    return_val: bool = False,
    eval_f1: bool = True,
    patience: int = 3,
    testing: bool = False,
    force_epochs=False,
    fast_val=False,
):
    """Function for model training.

    Args:
        n_epochs (int): Number of epochs used for model training.
        training_loader (DataLoader): Loader for the training set.
        validation_loader (DataLoader): Loader for the validation (dev) set.
        optimizer (torch.optim): Optimizer used for model training.
        scheduler (Callable): Scheduler used for learning rate.
        checkpoint_path (str): Path to store the model from the last epoch.
        best_model_path (str): Path to store the best model (wrt macro F score).

    Returns:
          The trained model.
    """
    # initialize tracker for minimum validation loss
    fscore_max = -np.Inf
    train_losses = []
    valid_losses = []
    valid_fscores = []
    last_loss = 0
    max_patience = patience
    epoch = 0
    max_epoch = n_epochs
    # print("WARNING! TEST CODE PRESENT IN transformers_multilabel_cap.py l.549")
    while (patience > 0) and (epoch < max_epoch):
        if force_epochs:
            if epoch > n_epochs:
                break
        train_loss = 0
        valid_loss = 0
        val_targets = []
        val_outputs = []
        model.train()
        print("############# Epoch {}: Training Start   #############".format(epoch))
        for batch_idx, data in enumerate(training_loader):
            # print('yyy epoch', batch_idx)
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = loss_fn(outputs, targets)
            train_loss = train_loss + loss.item()
            if batch_idx % 5 == 0:
                print(f"Epoch: {epoch}, Training Loss:  {loss.item()}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = train_loss

        if loss.item() < 0.5:
            print(
                "############# Epoch {}: Training End     #############".format(epoch)
            )

            print(
                "############# Epoch {}: Validation Start   #############".format(epoch)
            )
            ######################
            # validate the model #
            ######################
            model.eval()
            with torch.no_grad():
                for batch_idx, data in tqdm(
                    enumerate(validation_loader), total=len(validation_loader)
                ):
                    ids = data["ids"].to(device, dtype=torch.long)
                    mask = data["mask"].to(device, dtype=torch.long)
                    targets = data["labels"].to(device, dtype=torch.float)
                    outputs = model(ids, mask)

                    loss = loss_fn(outputs, targets)
                    valid_loss = valid_loss + loss.item()
                    val_targets.extend(targets.detach().cpu().numpy().tolist())
                    val_outputs.extend(
                        torch.sigmoid(outputs).detach().cpu().numpy().tolist()
                    )

                print(
                    "############# Epoch {}: Validation End     #############".format(
                        epoch
                    )
                )
            # calculate validation Fscore

            valid_loss = valid_loss / len(validation_loader)
            if eval_f1:
                best_cut = find_best_cut(val_outputs, val_targets)
                best_thr = best_cut["best_thr"]
                val_fscore = best_cut["best_f"]
            else:
                val_fscore = 1 - valid_loss
                best_thr = -1
            # calculate average losses
            # print('before cal avg train loss', train_loss)
            train_loss = train_loss / len(training_loader)
            # valid_loss = valid_loss / len(validation_loader)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_fscores.append(val_fscore)
            # print training/validation statistics
            print(
                "Epoch: {} \tAvgerage Training Loss: {:.6f} "
                "\tAverage Validation Loss: {:.6f} "
                "\tValidation F Score: {:.6f}".format(
                    epoch, train_loss, valid_loss, val_fscore
                )
            )

            # create checkpoint variable and add important data
            checkpoint = {
                "epoch": epoch + 1,
                "valid_loss": valid_loss,
                "val_fscore": val_fscore,
                "best_thr": best_thr,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            # save checkpoint
            # save_ckp(checkpoint, False, checkpoint_path, best_model_path)

            ## save the model that achieved the best macro F score
            if val_fscore >= fscore_max:
                print(
                    "Validation F score increased ({:.6f} --> {:.6f}).  Saving "
                    "model ...".format(fscore_max, val_fscore)
                )
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                fscore_max = val_fscore
                patience = max_patience
            else:
                patience -= 1
        else:
            pass
        epoch += 1
        # print(epoch)
        # if epoch > 20:
        #    break

    print("############# Epoch {}  Done   #############\n".format(epoch))
    if return_val:
        return (val_fscore, valid_loss)
    else:
        return model


def do_text_prediction(model: nn.Module, text: str):
    model.eval()


def do_prediction(loader: DataLoader, model: nn.Module):
    """Function for doing predictions for a given data loader and a model.

    Arguments:
        loader (DataLoader): Data loader for the dataset precitions are
        computer for.
        model (nn.Module): Model used for the predictions.

    Returns: List of output probabilities for the samples in the dataset and
    a list of their corresponding labels.
    """
    model.eval()
    fin_outputs = []
    fin_targets = []
    with torch.no_grad():
        for _, data in enumerate(loader):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.float)
            try:
                outputs = model(ids, mask)
            except:
                continue
            fin_targets.extend(targets.detach().cpu().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy().tolist())
    return fin_outputs, fin_targets


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function."""
    # get the data paths from config
    train_data_path = cfg.data.train_data_path
    val_data_path = cfg.data.val_data_path
    test_data_path = cfg.data.test_data_path
    # get the datasets in the proper format
    datasets = get_datasets(
        train_data_path, val_data_path, test_data_path, "comment_text"
    )
    # set parameters
    train_params = {
        "batch_size": cfg.experiment.batch_size,
        "shuffle": True,
        "num_workers": cfg.experiment.num_workers,
    }
    test_params = {
        "batch_size": cfg.experiment.batch_size,
        "shuffle": False,
        "num_workers": cfg.experiment.num_workers,
    }

    training_loader = DataLoader(
        datasets["train"], collate_fn=BertCollator(cfg.experiment.tok), **train_params
    )
    validation_loader = DataLoader(
        datasets["val"], collate_fn=BertCollator(cfg.experiment.tok), **test_params
    )
    testing_loader = DataLoader(
        datasets["test"], collate_fn=BertCollator(cfg.experiment.tok), **test_params
    )
    model = TransformersTextClassification(
        cfg.experiment.model, datasets["num_classes"]
    )

    model.to(device)
    # compute training and warmup steps needed for the scheduler
    num_training_steps = len(datasets["train"]) // cfg.experiment.batch_size

    num_warmup_steps = 0.06 * num_training_steps
    # get the optimizer and scheduler
    opt_sch = get_optimizer(
        model=model,
        lr=cfg.experiment.learning_rate,
        epochs=cfg.experiment.epochs,
        tr_steps=num_training_steps,
        wrm_steps=num_warmup_steps,
    )
    optimizer = opt_sch["optimizer"]
    scheduler = opt_sch["scheduler"]
    # load best model
    checkpoint_path = cfg.experiment.checkpoint_path
    best_model_path = cfg.experiment.best_model_path
    trained_model = train_model(
        n_epochs=cfg.experiment.epochs,
        training_loader=training_loader,
        validation_loader=validation_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
        best_model_path=best_model_path,
    )
    best_ckpt = load_ckp(best_model_path, model, optimizer)
    best_mod = best_ckpt["model"]
    thr_best = best_ckpt["thr"]
    # compute performance scores on the test set for the best model
    test_outputs, test_targets = do_prediction(testing_loader, best_mod)
    test_preds = (np.array(test_outputs) >= thr_best).astype(int)
    accuracy = metrics.accuracy_score(test_targets, test_preds)
    f1_score_micro = metrics.f1_score(test_targets, test_preds, average="micro")
    f1_score_macro = metrics.f1_score(test_targets, test_preds, average="macro")
    print(f"Test Accuracy Score = {accuracy}")
    print(f"Test F1 Score (Micro) = {f1_score_micro}")
    print(f"Test F1 Score (Macro) = {f1_score_macro}")


if __name__ == "__main__":
    main()
