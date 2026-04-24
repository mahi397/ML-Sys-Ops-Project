FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime


WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl git build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install llama-cpp-python with CUDA support separately
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python==0.2.56 --no-binary llama-cpp-python

# ADD THIS LINE — needed for MLflow S3 artifact download
RUN pip install --no-cache-dir boto3
##==0.2.56

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]