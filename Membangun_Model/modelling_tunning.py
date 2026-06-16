import itertools
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "preprocessed data" / "train.csv"
TEST_PATH = BASE_DIR / "preprocessed data" / "test.csv"
OUTPUT_DIR = BASE_DIR / "tuning_outputs"
BEST_DIR = OUTPUT_DIR / "best_checkpoint"


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_float_list(raw: str, default: list[float]) -> list[float]:
    if not raw.strip():
        return default
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    return [float(v) for v in vals]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_environment() -> dict:
    cfg = {
        "dagshub_owner": os.getenv("DAGSHUB_OWNER", ""),
        "dagshub_repo": os.getenv("DAGSHUB_REPO", ""),
        "dagshub_token": os.getenv("DAGSHUB_TOKEN", ""),
        "hf_model_name": os.getenv("HF_MODEL_NAME", "distilbert-base-uncased"),
        "hf_token": os.getenv("HF_TOKEN", ""),
        "experiment_name": os.getenv(
            "MLFLOW_EXPERIMENT_NAME_TUNING", "sml-transformer-finetuning-optimization"
        ),
        "registered_model_name": os.getenv(
            "MLFLOW_MODEL_NAME_TUNING", "stream-title-safety-classifier-finetuned"
        ),
        "batch_size": int(os.getenv("TRAIN_BATCH_SIZE", "16")),
        "eval_batch_size": int(os.getenv("EVAL_BATCH_SIZE", "32")),
        "max_length": int(os.getenv("MAX_LENGTH", "96")),
        "random_state": int(os.getenv("RANDOM_STATE", "42")),
        "num_epochs": int(os.getenv("NUM_EPOCHS", "3")),
        "learning_rates": parse_float_list(os.getenv("LEARNING_RATES", "2e-5,3e-5"), [2e-5, 3e-5]),
        "weight_decays": parse_float_list(os.getenv("WEIGHT_DECAYS", "0.0,0.01"), [0.0, 0.01]),
        "warmup_ratio": float(os.getenv("WARMUP_RATIO", "0.1")),
        "grad_clip_norm": float(os.getenv("GRAD_CLIP_NORM", "1.0")),
        "val_size": float(os.getenv("VALIDATION_SIZE", "0.15")),
    }

    missing = [k for k in ["dagshub_owner", "dagshub_repo", "dagshub_token"] if not cfg[k]]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing required environment variables (set via GitHub Secrets/Actions env): {joined}"
        )

    if not 0.0 < cfg["val_size"] < 1.0:
        raise ValueError("VALIDATION_SIZE must be in range (0,1)")

    return cfg


def configure_mlflow(cfg: dict) -> str:
    tracking_uri = f"https://dagshub.com/{cfg['dagshub_owner']}/{cfg['dagshub_repo']}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg["dagshub_owner"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg["dagshub_token"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(cfg["experiment_name"])

    return tracking_uri


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    required = {"clean_text", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    if df["clean_text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} contains null values in clean_text/label")


def build_dataloaders(
    tokenizer: AutoTokenizer,
    x_train: list[str],
    y_train: np.ndarray,
    x_val: list[str],
    y_val: np.ndarray,
    x_test: list[str],
    y_test: np.ndarray,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=max_length)

    train_dataset = TextClassificationDataset(train_encodings, y_train)
    val_dataset = TextClassificationDataset(val_encodings, y_val)
    test_dataset = TextClassificationDataset(test_encodings, y_test)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            losses.append(float(outputs.loss.item()))
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(inputs["labels"].cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, np.array(all_preds), np.array(all_labels)


def run_training_epoch(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []

    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def train_and_optimize(cfg: dict) -> None:
    set_seed(cfg["random_state"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    validate_dataframe(train_df, "train.csv")
    validate_dataframe(test_df, "test.csv")

    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_df["label"])
    test_labels = encoder.transform(test_df["label"])

    x_train_raw, x_val_raw, y_train_raw, y_val_raw = train_test_split(
        train_df["clean_text"].astype(str).tolist(),
        train_labels,
        test_size=cfg["val_size"],
        random_state=cfg["random_state"],
        stratify=train_labels,
    )

    hf_token = cfg["hf_token"] if cfg["hf_token"] else None
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model_name"], token=hf_token)

    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        x_train=x_train_raw,
        y_train=y_train_raw,
        x_val=x_val_raw,
        y_val=y_val_raw,
        x_test=test_df["clean_text"].astype(str).tolist(),
        y_test=test_labels,
        max_length=cfg["max_length"],
        train_batch_size=cfg["batch_size"],
        eval_batch_size=cfg["eval_batch_size"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = list(itertools.product(cfg["learning_rates"], cfg["weight_decays"]))

    best_trial = {
        "trial_id": None,
        "lr": None,
        "weight_decay": None,
        "best_epoch": None,
        "best_val_f1_weighted": -1.0,
    }

    config_path = OUTPUT_DIR / "tuning_config_used.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with mlflow.start_run(run_name="bert-finetuning-optimization") as run:
        mlflow.log_param("task", "binary_text_classification")
        mlflow.log_param("bert_usage", "full_fine_tuning")
        mlflow.log_param("autolog_used", "false")
        mlflow.log_param("hf_model_name", cfg["hf_model_name"])
        mlflow.log_param("train_batch_size", cfg["batch_size"])
        mlflow.log_param("eval_batch_size", cfg["eval_batch_size"])
        mlflow.log_param("max_length", cfg["max_length"])
        mlflow.log_param("random_state", cfg["random_state"])
        mlflow.log_param("num_epochs", cfg["num_epochs"])
        mlflow.log_param("val_size", cfg["val_size"])
        mlflow.log_param("learning_rates", ",".join(str(x) for x in cfg["learning_rates"]))
        mlflow.log_param("weight_decays", ",".join(str(x) for x in cfg["weight_decays"]))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("train_split_rows", len(x_train_raw))
        mlflow.log_param("val_split_rows", len(x_val_raw))

        for label_name, count in train_df["label"].value_counts().items():
            mlflow.log_param(f"train_count_{label_name}", int(count))

        mlflow.log_artifact(str(config_path), artifact_path="config")

        trial_counter = 0
        for lr, weight_decay in search_space:
            trial_counter += 1
            with mlflow.start_run(run_name=f"trial_{trial_counter}", nested=True):
                mlflow.log_param("trial_id", trial_counter)
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("weight_decay", weight_decay)

                model = AutoModelForSequenceClassification.from_pretrained(
                    cfg["hf_model_name"],
                    token=hf_token,
                    num_labels=len(encoder.classes_),
                    id2label={i: c for i, c in enumerate(encoder.classes_)},
                    label2id={c: i for i, c in enumerate(encoder.classes_)},
                ).to(device)

                optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                total_steps = len(train_loader) * cfg["num_epochs"]
                warmup_steps = int(total_steps * cfg["warmup_ratio"])
                scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )

                best_val_f1 = -1.0
                best_epoch = -1

                for epoch in range(1, cfg["num_epochs"] + 1):
                    train_loss = run_training_epoch(
                        model=model,
                        loader=train_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        device=device,
                        grad_clip_norm=cfg["grad_clip_norm"],
                    )

                    val_loss, val_preds, val_labels = evaluate_model(model, val_loader, device)
                    val_acc = accuracy_score(val_labels, val_preds)
                    val_f1_macro = f1_score(val_labels, val_preds, average="macro")
                    val_f1_weighted = f1_score(val_labels, val_preds, average="weighted")

                    mlflow.log_metric("train_loss", float(train_loss), step=epoch)
                    mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                    mlflow.log_metric("val_accuracy", float(val_acc), step=epoch)
                    mlflow.log_metric("val_f1_macro", float(val_f1_macro), step=epoch)
                    mlflow.log_metric("val_f1_weighted", float(val_f1_weighted), step=epoch)

                    if val_f1_weighted > best_val_f1:
                        best_val_f1 = float(val_f1_weighted)
                        best_epoch = epoch

                if best_val_f1 > best_trial["best_val_f1_weighted"]:
                    best_trial = {
                        "trial_id": trial_counter,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "best_epoch": best_epoch,
                        "best_val_f1_weighted": best_val_f1,
                    }
                    model.save_pretrained(BEST_DIR)
                    tokenizer.save_pretrained(BEST_DIR)

                mlflow.log_metric("best_val_f1_weighted", float(best_val_f1))
                mlflow.log_param("best_epoch", best_epoch)

        best_summary_path = OUTPUT_DIR / "best_trial_summary.json"
        with open(best_summary_path, "w", encoding="utf-8") as f:
            json.dump(best_trial, f, indent=2)
        mlflow.log_artifact(str(best_summary_path), artifact_path="optimization")

        best_model = AutoModelForSequenceClassification.from_pretrained(BEST_DIR).to(device)
        test_loss, test_preds, test_labels = evaluate_model(best_model, test_loader, device)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1_macro = f1_score(test_labels, test_preds, average="macro")
        test_f1_weighted = f1_score(test_labels, test_preds, average="weighted")
        test_report = classification_report(
            test_labels,
            test_preds,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        test_cm = confusion_matrix(test_labels, test_preds)

        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_f1_macro", float(test_f1_macro))
        mlflow.log_metric("test_f1_weighted", float(test_f1_weighted))

        for class_name in encoder.classes_:
            class_metrics = test_report.get(class_name)
            if class_metrics:
                mlflow.log_metric(f"test_precision_{class_name}", float(class_metrics["precision"]))
                mlflow.log_metric(f"test_recall_{class_name}", float(class_metrics["recall"]))
                mlflow.log_metric(f"test_f1_{class_name}", float(class_metrics["f1-score"]))

        report_path = OUTPUT_DIR / "test_classification_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(test_report, f, indent=2)

        cm_path = OUTPUT_DIR / "test_confusion_matrix.csv"
        pd.DataFrame(test_cm, index=encoder.classes_, columns=encoder.classes_).to_csv(cm_path)

        label_map_path = OUTPUT_DIR / "label_mapping.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({i: c for i, c in enumerate(encoder.classes_)}, f, indent=2)

        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
        mlflow.log_artifact(str(label_map_path), artifact_path="metadata")
        mlflow.log_artifacts(str(BEST_DIR), artifact_path="hf_checkpoint")

        model_info = mlflow.pytorch.log_model(
            pytorch_model=best_model,
            name="model",
            registered_model_name=cfg["registered_model_name"],
        )

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{cfg['registered_model_name']}'")
        latest_version = max(int(v.version) for v in versions) if versions else None

        mlflow.set_tag("tracking_provider", "dagshub")
        mlflow.set_tag("run_type", "fine_tuning_with_hyperparameter_optimization")
        mlflow.set_tag("hf_authenticated", str(bool(hf_token)).lower())
        mlflow.set_tag("best_trial_id", str(best_trial["trial_id"]))
        mlflow.set_tag("registered_model_name", cfg["registered_model_name"])
        if latest_version is not None:
            mlflow.set_tag("registered_model_version", str(latest_version))

        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        if latest_version is not None:
            print(f"Registered Model: {cfg['registered_model_name']} | Version: {latest_version}")
        print(
            f"Best Trial: {best_trial['trial_id']} | LR: {best_trial['lr']} | "
            f"WD: {best_trial['weight_decay']} | Val F1 Weighted: {best_trial['best_val_f1_weighted']:.4f}"
        )
        print(f"Test Accuracy: {test_acc:.4f} | Test F1 Weighted: {test_f1_weighted:.4f}")


def main() -> None:
    cfg = load_environment()
    tracking_uri = configure_mlflow(cfg)
    print(f"MLflow Tracking URI: {tracking_uri}")
    train_and_optimize(cfg)


if __name__ == "__main__":
    main()
