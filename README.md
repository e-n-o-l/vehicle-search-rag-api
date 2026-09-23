# Vehicle Search Multimodal RAG API Engine

A unified asynchronous API backend combining custom vision embedders, a Qdrant vector database, and a fine-tuned Qwen2.5 LLM into a complete Multimodal Retrieval-Augmented Generation (RAG) system.

### Architecture & Technical Workarounds
* **Pipeline Integration:** Orchestrates the customized DINOv2 Cross-Attention compression embedder, Qdrant vector retrieval engine, and fine-tuned Qwen2.5-7B LLM into a single execution graph.
* **Cross-Attention Single-Image Inference Trick:**
  * During ingestion, the Cross-Attention layer compresses a full multi-image gallery ($K$ images) into $N$ distinct centroid embeddings ($N=8$). However, passing a single user query image through Cross-Attention would skew its spatial coordinates, as the attention matrix depends on multi-image context.
  * **The Workaround:** Exploited the internal Softmax behavior in Cross-Attention—when a single query image is passed as a batch of independent tensors (reshaping $[1, K, D] \to [B, 1, D]$), Softmax weights degenerate into uniform identities. This causes the layer to output $N$ identical representations of the single input image without shifting its spatial vector coordinates, avoiding the need for a separate single-image encoder network.
* **Asynchronous Backend API:** Implemented in FastAPI using explicit service/repository interfaces and Pydantic schema validation.
* **Project Status:** *Architectural and algorithmic layout fully implemented. The module serves as a production blueprint; system integration, deployment, and operational testing remain in progress.*
