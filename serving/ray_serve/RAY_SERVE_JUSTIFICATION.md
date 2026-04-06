# Ray Serve Integration — Bonus Item

### The problem with FastAPI for multi-model pipelines

Our Jitsi recap system has a **two-stage pipeline**:
- **Stage A (RoBERTa):** ~8ms per request, handles 800 windows per meeting
- **Stage B (Mistral-7B):** ~4s per request, handles ~8 segments per meeting

With FastAPI, both models share a single process. This creates problems:

1. **No independent scaling:** Stage B is 500x slower than Stage A, but both
   run with the same number of workers. Under bursty traffic (5 meetings ending
   simultaneously), Stage B becomes the bottleneck while Stage A resources sit idle

2. **Resource contention:** The LLM's GPU memory usage can interfere with
   RoBERTa's inference when both run in the same process.

3. **No native batching:** FastAPI processes requests one at a time. To batch
   Stage A requests (like Triton does), we had to deploy a separate Triton server

### How Ray Serve solves these problems

1. **Independent deployment scaling:**
   ```python
   @serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.3})
   class SegmenterDeployment: ...

   @serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 0.7})
   class SummarizerDeployment: ...
   ```
   Each model is a separate deployment with its own replica count and GPU allocation.
   The segmenter gets 30% GPU, the summarizer gets 70% reflecting their actual
   resource needs. Under load, Ray Serve can auto-scale each independently

2. **Native request batching:**
   ```python
   @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.05)
   async def batch_predict(self, requests: list) -> list:
   ```
   Ray Serve's `@serve.batch` decorator automatically groups concurrent requests
   into batched GPU forward passes achieving the same benefit as Triton's dynamic
   batching without requiring a separate inference server or ONNX conversion

3. **Deployment composition:**
   ```python
   class RecapPipelineDeployment:
       def __init__(self, segmenter, summarizer):
           self.segmenter = segmenter
           self.summarizer = summarizer
   ```
   The full pipeline (Stage A to Stage B)  Ray Serve handles the internal routing and load balancing

4. **Resource isolation:**
   The segmenter and summarizer run in separate Ray actors with dedicated GPU
   memory allocations. The LLM cannot starve the segmenter of GPU resources

### Concrete comparison

| Feature | FastAPI | Triton | Ray Serve |
|---------|---------|--------|-----------|
| Multi-model pipeline | Manual HTTP calls | Separate server per model | Native composition |
| Request batching | Not built-in | Dynamic batching | `@serve.batch` decorator |
| Independent scaling | Separate containers | Separate model configs | Per-deployment replicas |
| GPU sharing | Single process | Separate containers | Fractional GPU allocation |
| Autoscaling | External (K8s) | External | Built-in per deployment |
| Python native | Yes | No (C++ server) | Yes |
| ONNX required | No | Yes | No |

### When to use each

- **FastAPI:** Simple single-model serving, development/prototyping
- **Triton:** Maximum throughput for a single model with dynamic batching
- **Ray Serve:** Multi-model pipelines where models need independent scaling
  and resource isolation like exactly our use case

### Expected performance

Ray Serve adds ~1-2ms overhead per request compared to raw FastAPI (due to Ray's
actor communication). For Stage A (8ms baseline), this is a ~15% increase.
However, under concurrent load, Ray Serve's batching should recover this overhead
and potentially exceed FastAPI throughput similar to how Triton's batching
improved throughput from 123 req/s to 337 req/s

The real benefit is operational: a single Ray Serve deployment replaces both
FastAPI + Triton, simplifies the Docker setup, and provides autoscaling
without Kubernetes
