import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "preprocessed data" / "train.csv"
TEST_PATH = BASE_DIR / "preprocessed data" / "test.csv"


def load_environment() -> dict:
	cfg = {
		"dagshub_owner": os.getenv("DAGSHUB_OWNER", ""),
		"dagshub_repo": os.getenv("DAGSHUB_REPO", ""),
		"dagshub_token": os.getenv("DAGSHUB_TOKEN", ""),
		"hf_token": os.getenv("HF_TOKEN", ""),
		"experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "sml-transformer-feature-extraction"),
		"registered_model_name": os.getenv("MLFLOW_MODEL_NAME", "stream-title-safety-classifier"),
		"hf_model_name": os.getenv("HF_MODEL_NAME", "distilbert-base-uncased"),
		"sklearn_serialization_format": os.getenv("SKLEARN_SERIALIZATION_FORMAT", "skops"),
		"batch_size": int(os.getenv("BATCH_SIZE", "32")),
		"max_length": int(os.getenv("MAX_LENGTH", "96")),
		"random_state": int(os.getenv("RANDOM_STATE", "42")),
	}

	missing = [k for k in ["dagshub_owner", "dagshub_repo", "dagshub_token"] if not cfg[k]]
	if missing:
		joined = ", ".join(missing)
		raise ValueError(f"Missing required environment variables (set via GitHub Secrets/Actions env): {joined}")

	return cfg


def configure_mlflow(cfg: dict) -> str:
	tracking_uri = f"https://dagshub.com/{cfg['dagshub_owner']}/{cfg['dagshub_repo']}.mlflow"

	os.environ["MLFLOW_TRACKING_USERNAME"] = cfg["dagshub_owner"]
	os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg["dagshub_token"]

	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_registry_uri(tracking_uri)
	mlflow.set_experiment(cfg["experiment_name"])
	# Keep manual logging active while adding automatic sklearn logs.
	mlflow.sklearn.autolog(log_models=False, exclusive=False, silent=True)

	return tracking_uri


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
	required = {"clean_text", "label"}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"{name} missing required columns: {sorted(missing)}")

	if df["clean_text"].isna().any() or df["label"].isna().any():
		raise ValueError(f"{name} contains null values in clean_text/label")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
	masked = last_hidden_state * mask
	summed = masked.sum(dim=1)
	counts = torch.clamp(mask.sum(dim=1), min=1e-9)
	return summed / counts


def build_embeddings(
	texts: list[str],
	tokenizer: AutoTokenizer,
	model: AutoModel,
	batch_size: int,
	max_length: int,
	device: torch.device,
) -> np.ndarray:
	vectors = []
	model.eval()

	for start in range(0, len(texts), batch_size):
		batch = texts[start : start + batch_size]
		tokens = tokenizer(
			batch,
			padding=True,
			truncation=True,
			max_length=max_length,
			return_tensors="pt",
		)
		tokens = {k: v.to(device) for k, v in tokens.items()}

		with torch.no_grad():
			outputs = model(**tokens)
			pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
			vectors.append(pooled.cpu().numpy())

	return np.vstack(vectors)


def train_and_evaluate(cfg: dict) -> None:
	train_df = pd.read_csv(TRAIN_PATH)
	test_df = pd.read_csv(TEST_PATH)

	validate_dataframe(train_df, "train.csv")
	validate_dataframe(test_df, "test.csv")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	hf_token = cfg["hf_token"] if cfg["hf_token"] else None
	tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model_name"], token=hf_token)
	model = AutoModel.from_pretrained(cfg["hf_model_name"], token=hf_token).to(device)

	x_train_text = train_df["clean_text"].astype(str).tolist()
	x_test_text = test_df["clean_text"].astype(str).tolist()

	x_train = build_embeddings(
		texts=x_train_text,
		tokenizer=tokenizer,
		model=model,
		batch_size=cfg["batch_size"],
		max_length=cfg["max_length"],
		device=device,
	)
	x_test = build_embeddings(
		texts=x_test_text,
		tokenizer=tokenizer,
		model=model,
		batch_size=cfg["batch_size"],
		max_length=cfg["max_length"],
		device=device,
	)

	encoder = LabelEncoder()
	y_train = encoder.fit_transform(train_df["label"])
	y_test = encoder.transform(test_df["label"])

	classifier = LogisticRegression(
		random_state=cfg["random_state"],
		max_iter=2000,
		class_weight="balanced",
	)
	classifier.fit(x_train, y_train)
	y_pred = classifier.predict(x_test)

	accuracy = accuracy_score(y_test, y_pred)
	f1_macro = f1_score(y_test, y_pred, average="macro")
	f1_weighted = f1_score(y_test, y_pred, average="weighted")
	report = classification_report(
		y_test,
		y_pred,
		target_names=encoder.classes_,
		output_dict=True,
		zero_division=0,
	)

	pipeline = Pipeline([("classifier", classifier)])

	with mlflow.start_run(run_name="frozen-transformer-features") as run:
		mlflow.log_param("task", "binary_text_classification")
		mlflow.log_param("bert_usage", "feature_extraction_only_no_fine_tuning")
		mlflow.log_param("hf_model_name", cfg["hf_model_name"])
		mlflow.log_param("batch_size", cfg["batch_size"])
		mlflow.log_param("max_length", cfg["max_length"])
		mlflow.log_param("random_state", cfg["random_state"])
		mlflow.log_param("classifier", "LogisticRegression")
		mlflow.log_param("class_weight", "balanced")
		mlflow.log_param("train_rows", len(train_df))
		mlflow.log_param("test_rows", len(test_df))

		for label_name, count in train_df["label"].value_counts().items():
			mlflow.log_param(f"train_count_{label_name}", int(count))

		mlflow.log_metric("accuracy", float(accuracy))
		mlflow.log_metric("f1_macro", float(f1_macro))
		mlflow.log_metric("f1_weighted", float(f1_weighted))

		for class_name in encoder.classes_:
			if class_name in report:
				class_metrics = report[class_name]
				mlflow.log_metric(f"precision_{class_name}", float(class_metrics["precision"]))
				mlflow.log_metric(f"recall_{class_name}", float(class_metrics["recall"]))
				mlflow.log_metric(f"f1_{class_name}", float(class_metrics["f1-score"]))

		report_path = BASE_DIR / "classification_report.json"
		with open(report_path, "w", encoding="utf-8") as f:
			json.dump(report, f, indent=2)
		mlflow.log_artifact(str(report_path), artifact_path="metrics")

		mapping_path = BASE_DIR / "label_mapping.json"
		with open(mapping_path, "w", encoding="utf-8") as f:
			json.dump({i: c for i, c in enumerate(encoder.classes_)}, f, indent=2)
		mlflow.log_artifact(str(mapping_path), artifact_path="metadata")

		input_example = pd.DataFrame(x_test[:5])
		signature = infer_signature(pd.DataFrame(x_train[:10]), pipeline.predict(x_train[:10]))
		try:
			model_info = mlflow.sklearn.log_model(
				sk_model=pipeline,
				name="model",
				registered_model_name=cfg["registered_model_name"],
				input_example=input_example,
				signature=signature,
				serialization_format=cfg["sklearn_serialization_format"],
			)
		except Exception as exc:
			# Fallback to cloudpickle if skops is unavailable in runtime.
			mlflow.set_tag("model_log_fallback", f"cloudpickle_due_to_{type(exc).__name__}")
			model_info = mlflow.sklearn.log_model(
				sk_model=pipeline,
				name="model",
				registered_model_name=cfg["registered_model_name"],
				input_example=input_example,
				signature=signature,
				serialization_format="cloudpickle",
			)

		client = MlflowClient()
		versions = client.search_model_versions(f"name='{cfg['registered_model_name']}'")
		latest_version = max(int(v.version) for v in versions) if versions else None

		mlflow.set_tag("tracking_provider", "dagshub")
		mlflow.set_tag("run_type", "train_without_fine_tuning")
		mlflow.set_tag("mlflow.runName", "frozen-transformer-features")
		mlflow.set_tag("hf_authenticated", str(bool(hf_token)).lower())
		mlflow.set_tag("sklearn_serialization_format", cfg["sklearn_serialization_format"])
		if latest_version is not None:
			mlflow.set_tag("registered_model_version", str(latest_version))

		print(f"Run ID: {run.info.run_id}")
		print(f"Model URI: {model_info.model_uri}")
		if latest_version is not None:
			print(
				f"Registered Model: {cfg['registered_model_name']} | Version: {latest_version}"
			)
		print(f"Accuracy: {accuracy:.4f} | F1 Weighted: {f1_weighted:.4f}")


def main() -> None:
	cfg = load_environment()
	tracking_uri = configure_mlflow(cfg)
	print(f"MLflow Tracking URI: {tracking_uri}")
	train_and_evaluate(cfg)


if __name__ == "__main__":
	main()
