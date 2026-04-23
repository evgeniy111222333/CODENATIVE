# HTM Code-Native

Phases A, B, and C of the `HTM_Code_Native_Final_Concept.md` architecture.

This repository bootstraps a working vertical slice for:

- code-aware Python tokenization with byte alignment
- AST and symbol enrichment
- structural boundary scheduling
- code/byte/structure encoders
- hierarchical state-space updates (HSSM)
- semantic hot/cold memory
- exact recent memory (ERM)
- exact episodic memory (EEM)
- fused LM/copy output and smoke-train loop

## Layout

```text
configs/
docs/
benchmarks/
src/htm_code_native/
tests/
```

## Quick Start

```bash
python -m htm_code_native.cli tokenize tests/fixtures/sample_module.py
python -m htm_code_native.cli inspect tests/fixtures/sample_module.py
python -m htm_code_native.cli run-forward tests/fixtures/episodic_copy_module.py
python -m htm_code_native.cli smoke-train
python benchmarks/microbench.py tests/fixtures/episodic_copy_module.py
```

## Status

The repository now includes:

- Phase A semantic core
- Phase B exact recent memory
- Phase C exact episodic memory

Still deferred:

- repository graph memory (RGM)
- learned retrieval router

The main executable path is `PythonTokenizer -> PythonStructureExtractor -> BoundaryScheduler -> PhaseACodeModel`.
