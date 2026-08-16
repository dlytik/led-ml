# LED-ML Roadmap

# Explanation of current slowness
1 use_cache=False
2 run_model runs model(...) for every new token from start, forward pass repeats over and over

## v0.2 - Performance
- [ ] **Forward pass repeats for token 0
- optimize no_grad
- [ ] **KV-Cached Backward Pass**: Current impl is O(T^2). Implement gradient checkpointing over cached hidden_states to enable linear O(T) decoding while preserving per-token VJP tracing.
- [ ] **Per-layer ten_pct_count**: Move `ten_pct_count` calculation inside hook for variable hidden_dim models like MoE.

## v0.3 - Features  
- [ ] **Distributed TP Support**: Replace warning with `dist.all_gather()` for global neuron index mapping.
- [ ] **Export Formats**: Add JSONL and TensorBoard logging for trajectory data.

## Known Limitations
- DECISION mode still slow due to lack of KV cache. This is architectural, not a bug.
