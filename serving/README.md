# Serving 

## Structure

```
serving
├── app
│   ├── config.py          # Configuration settings for the application
│   ├── llm.py             # Functions and classes for LLM operations
│   ├── main.py            # Entry point of the application
│   ├── model.py           # Model loading and inference handling
│   ├── model_onnx.py      # Functions for working with ONNX models
│   ├── schemas.py         # Data schemas for input and output validation
│   └── tokenize.py        # Functions for tokenizing input text
├── benchmark
│   └── benchmark.py       # Benchmarking code for model performance evaluation
├── models                 # Directory for storing model files (currently empty)
├── scripts
│   └── export_onnx.py     # Scripts for exporting models to ONNX format
├── triton_models
│   └── roberta_segmenter
│       └── 1              # Versioned Triton Inference Server model files
├── Dockerfile              # Instructions for building the Docker image
├── docker-compose.yml      # Defines services for running the application
├── requirements.txt        # Lists Python dependencies for the project
├── .gitignore              # Specifies files to be ignored by Git
└── README.md               # Documentation for the project
```


- The application serves a large language model and provides endpoints for inference

