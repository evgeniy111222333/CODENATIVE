# Phase A-D Overview

The current repository implements four layers of the HTM Code-Native design:

1. Python source is tokenized with byte-accurate spans.
2. AST and symbol metadata enrich each token.
3. A structural scheduler emits level boundaries for HSSM updates.
4. Code, byte, and structural encoders create fused token embeddings.
5. HSSM builds hierarchical hidden states across levels `0..5`.
6. Semantic memory reads and writes hot/cold semantic slots.
7. Exact recent memory adds short-range copy distribution.
8. Exact episodic memory stores immutable chunks and long-range pointer copy.
9. Repository graph memory indexes Python, TS/JS, config, test, and diagnostic nodes for repo-scope retrieval.
10. A fused hidden state produces blended LM + semantic + recent + episodic + graph logits.

What is intentionally deferred:

- learned retrieval router
