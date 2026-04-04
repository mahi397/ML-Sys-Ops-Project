FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Python deps — pinned for reproducibility
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.19.0 \
    tokenizers==0.19.1 \
    scikit-learn==1.4.2 \
    mlflow==2.13.0 \
    boto3==1.34.0 \
    optuna==3.6.1 \
    seqeval==1.2.2 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    accelerate==0.30.0 \
    nltk==3.8.1 \
    "ray[train,default]==2.10.0"

# Download NLTK data at build time so workers don't need internet access at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . /app

ENTRYPOINT ["python"]
