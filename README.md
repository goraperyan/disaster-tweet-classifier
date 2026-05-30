# Disaster Tweet Classifier

Automatic classification of disaster-related tweets using a complete lightweight MLOps pipeline.

**Author:** Gor Aperyan

## 1. Project Overview

This project solves a binary text classification task: given a short English tweet from Twitter/X, the system predicts whether the tweet describes a real disaster or not.

The project is based on the Kaggle competition **Natural Language Processing with Disaster Tweets**:

- Competition page: <https://www.kaggle.com/competitions/nlp-getting-started>
- Dataset page: <https://www.kaggle.com/competitions/nlp-getting-started/data>

The goal of the project is not to achieve state-of-the-art leaderboard performance, but to demonstrate a complete and reproducible MLOps workflow:

- data versioning with DVC;
- configurable training with Hydra;
- baseline model training;
- neural model training with PyTorch Lightning;
- experiment tracking with MLflow;
- model inference;
- Kaggle submission generation;
- FastAPI serving;
- Docker packaging;
- tests and pre-commit checks.

## 2. Task Definition

The task is binary classification of short English tweets.

For each input tweet, the model predicts whether the message is related to a real disaster.

Target labels:

- `0`: not a real disaster;
- `1`: real disaster.

The model output is a probability of the positive class. A binary class is then obtained by thresholding this probability.

By default, the threshold is `0.5`.

Post-processing rule:

- if `probability >= 0.5`, then `label = 1`;
- otherwise, `label = 0`.

## 3. Dataset

The project uses the Kaggle dataset **Natural Language Processing with Disaster Tweets**.

Dataset source:

<https://www.kaggle.com/competitions/nlp-getting-started/data>

The dataset was published by Kaggle as part of the `nlp-getting-started` competition. It is a small supervised NLP dataset intended for introductory natural language processing and text classification experiments.

### 3.1 Dataset Files

The competition provides the following files:

- `train.csv`: labeled training data.
- `test.csv`: unlabeled test data for Kaggle submission.
- `sample_submission.csv`: submission format example.

Approximate dataset size:

- `train.csv`: 7,613 rows, contains labels, less than 1 MB.
- `test.csv`: 3,263 rows, does not contain labels, less than 1 MB.
- `sample_submission.csv`: 3,263 rows, template only, less than 1 MB.

The full raw dataset is small and is approximately a few megabytes on disk.

### 3.2 Input Fields

Each object in the dataset contains the following fields:

- `id`: unique tweet identifier.
- `keyword`: keyword associated with the tweet; can be missing.
- `location`: user-provided location string; can be missing.
- `text`: tweet text; the main input field used by the model.
- `target`: binary target label, available only in `train.csv`.

The main model input is the `text` field.

### 3.3 Example Data

Example positive tweet:

- Text: `Forest fire near La Ronge Sask. Canada`
- Target: `1`

Example negative tweet:

- Text: `What's up man?`
- Target: `0`

### 3.4 Text Length

Tweets are short texts. In this project, the neural model uses a maximum sequence length of `128` tokens.

This is enough for the majority of tweets in the dataset and keeps training and inference computationally lightweight.

The raw text length is limited by the nature of Twitter/X posts and usually fits comfortably within this token limit.

### 3.5 Data Challenges

The dataset is small and noisy. Main difficulties include:

- short texts with limited context;
- informal language;
- abbreviations and slang;
- hashtags;
- mentions;
- URLs;
- punctuation noise;
- emojis and special symbols;
- metaphorical language, for example words such as `fire`, `storm`, or `crash` used in a non-disaster meaning;
- missing values in `keyword` and `location`;
- possible duplicate or near-duplicate tweets;
- moderate class imbalance;
- hidden test labels on Kaggle.

Because the dataset contains only about 7.6k labeled examples, overfitting is a risk for large transformer models. To reduce this risk, the project uses pretrained language models, validation monitoring, early stopping, and a small number of epochs.

## 4. Metrics

The main metric is **F1-score**, because both precision and recall are important for this task.

The project tracks the following metrics:

- F1-score: main metric; balances precision and recall.
- Precision: measures how many predicted disasters are real disasters.
- Recall: measures how many real disasters were found.
- Accuracy: general classification correctness.
- ROC-AUC: ranking quality of predicted probabilities.
- Loss: training and validation optimization objective.

Expected metric ranges:

- TF-IDF + Logistic Regression baseline: `0.70–0.75` F1-score.
- BERTweet-based model: `0.80–0.85` F1-score.

These values are realistic for this dataset because:

- the dataset is small;
- the texts are noisy and short;
- the Kaggle public leaderboard for this introductory competition typically contains many solutions in the approximate `0.78–0.84` F1 range;
- transformer models pretrained on tweets usually outperform classical TF-IDF baselines on noisy social media text;
- the project intentionally prioritizes reproducibility and MLOps completeness over aggressive leaderboard tuning.

## 5. Validation and Test Strategy

The project uses stratified validation.

During data preparation, the training data is split into folds using stratification by the target label. This helps preserve the class distribution between training and validation subsets.

The current training pipeline uses a selected validation fold for model selection and metric reporting.

The hidden Kaggle test set is used only for final submission generation. Since true labels for `test.csv` are not available locally, local test metrics cannot be computed on the Kaggle test set.

The Kaggle submission pipeline uses:

- `test.csv` as input;
- the trained model checkpoint;
- `sample_submission.csv` as the required output format template.

## 6. Modeling

The project contains two model families:

1. a classical lightweight baseline;
2. a transformer-based neural model.

### 6.1 Baseline Model

The baseline model consists of:

- text preprocessing;
- TF-IDF vectorization;
- Logistic Regression classifier.

Baseline preprocessing includes:

- lowercasing;
- URL normalization/removal depending on configuration;
- mention normalization/removal depending on configuration;
- hashtag processing;
- punctuation cleanup;
- whitespace normalization.

The baseline is intentionally simple and CPU-friendly. It provides a reference point for evaluating the benefit of the neural model.

### 6.2 Main Model

The main model is based on **BERTweet**.

BERTweet is selected because it is pretrained on Twitter text and is therefore suitable for short, noisy tweets containing hashtags, mentions, slang, and informal constructions.

The model architecture:

- pretrained BERTweet encoder;
- classification head;
- binary classification output.

Training framework:

- PyTorch;
- PyTorch Lightning;
- Hydra for configuration;
- MLflow for experiment tracking.

Approximate training configuration:

- Maximum sequence length: `128`.
- Batch size: `16`.
- Optimizer: `AdamW`.
- Learning rate: around `2e-5`.
- Loss function: cross entropy loss.
- Epochs: `2–4`.
- Model selection: best validation F1-score.

The project uses Hugging Face components for pretrained tokenizers and model backbones, but the training loop is implemented with PyTorch Lightning rather than Hugging Face Trainer.

## 7. Model Format and Artifacts

The PyTorch Lightning model is saved as a checkpoint:

`artifacts/models/bertweet/checkpoints/`

The baseline model is saved as a serialized artifact:

`artifacts/models/baseline/`

Additional artifacts include:

- `artifacts/models/bertweet/metrics.json`
- `artifacts/submissions/bertweet_submission.csv`
- `plots/`
- `mlruns/`

The project also contains configuration for ONNX export:

`artifacts/models/bertweet/onnx/model.onnx`

The primary serving path currently uses the PyTorch checkpoint. ONNX export is included as a production-oriented extension.

## 8. Inference and Deployment

The trained model is served as a REST API using FastAPI.

The service contains:

- model loading;
- tokenizer initialization;
- preprocessing/tokenization;
- probability prediction;
- threshold-based post-processing;
- JSON response formatting.

### 8.1 API Endpoints

The API contains the following endpoints:

- `GET /health`: service health check.
- `GET /model-info`: current model metadata.
- `POST /predict`: predict label for a single tweet.

### 8.2 Example Prediction Request

Run:

    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{"text": "Forest fire near La Ronge Sask. Canada"}'

Example response:

    {
      "text": "Forest fire near La Ronge Sask. Canada",
      "label": 1,
      "label_name": "disaster",
      "probability": 0.91,
      "threshold": 0.5
    }

### 8.3 Inference Resources

The service is designed to run on CPU for demonstration purposes.

Recommended local inference setup:

- CPU: 2 or more cores.
- RAM: 4 GB or more.
- GPU: not required for inference.
- Batch size: `1` for REST API.
- Expected latency: suitable for demo and small-scale usage.

For higher throughput production inference, possible extensions include:

- batching;
- ONNX Runtime;
- TorchScript;
- TensorRT;
- Triton Inference Server;
- horizontal scaling of the FastAPI service.

## 9. Project Structure

    .
    ├── configs/                         # Hydra configuration files
    │   ├── data/
    │   ├── export/
    │   ├── inference/
    │   ├── logging/
    │   ├── model/
    │   ├── preprocessing/
    │   ├── serving/
    │   └── training/
    ├── data/                            # Raw and processed data, tracked with DVC
    ├── disaster_tweet_classifier/        # Main Python package
    │   ├── api/                          # FastAPI application and schemas
    │   ├── cli/                          # CLI commands
    │   ├── data/                         # Data preparation utilities
    │   ├── inference/                    # Inference services
    │   ├── models/                       # Model definitions
    │   ├── preprocessing/                # Text preprocessing
    │   ├── tracking/                     # MLflow and git tracking utilities
    │   ├── training/                     # Training logic and metrics
    │   └── utils/                        # Shared utilities
    ├── tests/                            # Unit and integration tests
    ├── artifacts/                        # Model artifacts and generated outputs
    ├── dvc.yaml                          # DVC pipeline definition
    ├── pyproject.toml                    # Project dependencies and tooling
    ├── uv.lock                           # Locked dependency versions
    ├── Dockerfile                        # Docker image for API serving
    ├── .dockerignore
    ├── .pre-commit-config.yaml
    └── README.md

## 10. Quick Check for Reviewers

Minimal verification sequence:

    git clone git@github.com:goraperyan/disaster-tweet-classifier.git
    cd disaster-tweet-classifier
    uv sync
    uv run pre-commit install
    uv run pre-commit run -a
    uv run pytest
    uv run disaster-tweet train-baseline

If the Kaggle raw data and trained artifacts are available through a DVC remote:

    uv run dvc pull
    uv run disaster-tweet train-bertweet
    uv run disaster-tweet serve-api

## 11. Setup

This project uses `uv` for dependency management.

### 11.1 Clone Repository

    git clone git@github.com:goraperyan/disaster-tweet-classifier.git
    cd disaster-tweet-classifier

### 11.2 Install uv

If `uv` is not installed:

    pip install uv

Alternatively, follow the official installation instructions:

<https://docs.astral.sh/uv/>

### 11.3 Create Environment and Install Dependencies

    uv sync

This creates a virtual environment and installs all project dependencies from `uv.lock`.

### 11.4 Activate Environment

You can either run commands through `uv run`:

    uv run python --version

or activate the environment manually:

    source .venv/bin/activate

### 11.5 Install Pre-commit Hooks

    uv run pre-commit install

Run all pre-commit checks:

    uv run pre-commit run -a

Expected result: all checks pass.

## 12. Data Setup

The raw Kaggle data should be placed under:

`data/raw/`

Expected files:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

The raw data is not tracked directly by Git. It is managed with DVC.

If a DVC remote is configured, restore data and artifacts with:

    uv run dvc pull

If no DVC remote is available, download the dataset manually from Kaggle:

<https://www.kaggle.com/competitions/nlp-getting-started/data>

and place the files into `data/raw/`.

## 13. DVC Pipeline

Show the pipeline graph:

    uv run dvc dag

Check pipeline status:

    uv run dvc status

Reproduce the full available pipeline:

    uv run dvc repro

Typical stages include:

- fold preparation;
- text preprocessing;
- baseline data preparation;
- baseline training;
- BERTweet training;
- submission generation.

## 14. Training

### 14.1 Train Baseline Model

    uv run disaster-tweet train-baseline

The baseline model trains a TF-IDF + Logistic Regression classifier and saves metrics and artifacts.

### 14.2 Train BERTweet Model

    uv run disaster-tweet train-bertweet

Expected behavior:

- the training starts successfully;
- training and validation loss are logged;
- validation metrics are computed;
- the best checkpoint is saved;
- MLflow logs metrics, parameters, and artifacts.

The model is selected by validation F1-score.

Example expected F1-score range:

`0.80–0.85`

The exact value can vary depending on hardware, random seed, dependency versions, and fold split.

## 15. Experiment Tracking with MLflow

MLflow is used for experiment tracking.

Depending on configuration, MLflow logs are written to:

`mlruns/`

or to a configured tracking server.

To start a local MLflow UI:

    uv run mlflow ui --backend-store-uri mlruns

Then open:

`http://127.0.0.1:5000`

Logged information includes:

- parameters;
- metrics;
- model artifacts;
- plots;
- git commit hash where available.

## 16. Kaggle Submission

Generate a Kaggle submission file:

    uv run disaster-tweet predict-submission

Expected output:

`artifacts/submissions/bertweet_submission.csv`

The file follows the format of `sample_submission.csv` and can be submitted to Kaggle.

## 17. FastAPI Inference Service

Run the API locally:

    uv run disaster-tweet serve-api

or directly with Uvicorn:

    uv run uvicorn disaster_tweet_classifier.api.app:app --host 0.0.0.0 --port 8000

Health check:

    curl http://localhost:8000/health

Model information:

    curl http://localhost:8000/model-info

Single prediction:

    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{"text": "Forest fire near La Ronge Sask. Canada"}'

## 18. Docker

Build the Docker image:

    docker build -t disaster-tweet-classifier:latest .

Run the API service:

    docker run --rm -p 8000:8000 disaster-tweet-classifier:latest

Test the containerized API:

    curl http://localhost:8000/health

    curl http://localhost:8000/model-info

    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{"text": "Forest fire near La Ronge Sask. Canada"}'

The Docker image copies the trained BERTweet artifacts from:

`artifacts/models/bertweet/`

Therefore, the model artifacts should exist before building the image.

## 19. Testing

Run all tests:

    uv run pytest

Run API tests only:

    uv run pytest tests/api -q

Run pre-commit checks:

    uv run pre-commit run -a

## 20. Code Quality

The project uses pre-commit hooks for automatic quality checks.

The exact hooks are defined in:

`.pre-commit-config.yaml`

Typical checks include:

- formatting;
- linting;
- import sorting;
- static checks;
- YAML/TOML validation;
- trailing whitespace removal;
- end-of-file fixing.
