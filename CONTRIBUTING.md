# Contributing to LED-ML

Thanks for your interest in improving LED-ML. This project is still evolving, and the roadmap is intentionally focused on making runtime diagnostics faster, more scalable, and more useful for real-world LLM workloads.

## Project goals

LED-ML is built to provide low-overhead structural telemetry for large language models. The current roadmap in `ISSUES.md` highlights the biggest opportunities:

- reduce repeated forward-pass work during token-by-token decoding
- improve efficiency with KV-cached backward behavior
- make per-layer metric sizing robust across variable hidden dimensions
- add distributed tensor-parallel support
- expand output formats beyond JSON snapshots

This is a research-oriented, performance-sensitive codebase, so contributions that improve correctness, throughput, and maintainability are especially welcome.

## Current priorities

The project currently tracks the following areas:

### v0.2 - Performance
- fix repeated forward pass behavior for token 0
- reduce overhead using `no_grad` where appropriate
- implement KV-cached backward pass logic to avoid O(T^2) behavior
- move `ten_pct_count` logic into the hook path for variable hidden dimensions such as MoE models

### v0.3 - Features
- add distributed TP support using global neuron mapping instead of warnings
- add JSONL and TensorBoard export formats for trajectory data

### Known limitations
- decision mode remains slower because it still lacks an effective KV cache
- this is architectural and should be treated as a design constraint, not as a bug report against the overall approach

## How to contribute

We welcome contributions in several forms:

- performance tuning and benchmarking
- correctness fixes for model hooks and telemetry extraction
- distributed inference support
- new export/logging formats
- documentation and examples
- reproducible benchmark scripts and measurements

## Suggested workflow

1. Review `ISSUES.md` and choose an item that matches your interest or expertise.
2. Open or comment on an issue before starting major work if the change is non-trivial.
3. Fork the repository and create a focused branch.
4. Keep the patch narrow and explain the motivation in your PR description.
5. Validate behavior with the smallest reproduction or benchmark you can run.
6. Open a pull request with:
   - a summary of the change
   - the issue or roadmap item it addresses
   - any performance or correctness evidence

## Good contribution types

We especially value contributions that:

- measure before and after impact
- improve runtime or memory efficiency without changing semantics
- make model compatibility broader and more robust
- reduce uncertainty for unsupported or partially supported architectures
- improve documentation for setup, usage, and roadmap execution

## Development guidance

- Prefer small, readable changes over broad rewrites.
- Keep diagnostics and telemetry logic understandable and easy to benchmark.
- Be careful with model hooks, tensor shapes, and distributed execution assumptions.
- Document assumptions clearly when the change depends on a specific architecture or runtime environment.

## Collaboration call

We are actively looking for collaborators across the following areas:

- LLM runtime optimization
- distributed training/inference support
- tensor and activation analysis
- benchmarking and profiling
- tooling for export and visualization

If you are working on similar problems or want to help move this project toward a faster, more scalable v0.2/v0.3 roadmap, we would love to collaborate.

Please open an issue, start a discussion, or submit a PR. The project benefits most from contributors who can pair a concrete improvement with measured evidence and a clear explanation of the tradeoff.

Thank you for helping build the next iteration of LED-ML.
