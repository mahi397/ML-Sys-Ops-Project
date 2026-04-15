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
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging
# install llama-cpp-python with CUDA support separately
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python 
##==0.2.56

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]