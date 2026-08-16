import torch
from typing import Dict, Any, List
import json
import torch.distributed as dist
import os
import warnings
import time
from importlib import resources
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from .constants import POS_LED, NEG_LED, DEAD_LED, ALIVE_LED

from enum import Enum

# =====================================================================
# SYSTEM UTILITIES (DISTRIBUTED & BENCHMARKING)
# =====================================================================
class OutputType(Enum):
    DECISION = "decision"
    TRAJECTORY = "trajectory"

class LoopState:
    def __init__(self):
        self.step = 0

def _is_distributed():
    return dist.is_available() and dist.is_initialized()

def _get_rank():
    return dist.get_rank() if _is_distributed() else 0

def _get_world_size():
    return dist.get_world_size() if _is_distributed() else 1

def get_nested_attr(obj, attr_path):
    """Safely resolves nested strings like 'self_attn.o_proj'"""
    parts = attr_path.split('.')
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj
class led_core:

    def __init__(self, mode=None):
        if torch.cuda.is_available():
            self._sync_fn = torch.cuda.synchronize
            self._time_fn = time.perf_counter  # High-resolution CPU clock paired with sync
        elif hasattr(torch, "backends") and torch.backends.mps.is_available():
            self._sync_fn = torch.mps.synchronize
            self._time_fn = time.perf_counter
        else:
            self._sync_fn = lambda: None  # Fallback for plain CPU execution
            self._time_fn = time.perf_counter

        self._layer_metrics = {}
        self.handles = []
        self.num_layers = 0
        self.config = self._load_config()
        self.mode = OutputType.DECISION if mode is None else OutputType(mode)
        self.loop_state = LoopState()
        self.num_tokens = 0
        self.ten_pct_count = 0
    

    def _load_config(self) -> Dict[str, Any]:
        """Load supported models configuration from package resource."""
        try:
            # Python 3.9+: Use importlib.resources.files
            if hasattr(resources, 'files'):
                from importlib.resources import files
                config_file = files('led_ml').joinpath('supported_models.json')
                config_data = config_file.read_text(encoding='utf-8')
            else:
                # Fallback for Python 3.7-3.8
                with resources.open_text('led_ml', 'supported_models.json') as f:
                    config_data = f.read()
            
            return json.loads(config_data)
        except (FileNotFoundError, ModuleNotFoundError):
            # Fallback: read from file system if resource not found
            from pathlib import Path
            config_path = Path(__file__).parent / 'supported_models.json'
            with open(config_path, 'r') as f:
                return json.load(f)
    
    def get_supported_models(self) -> Dict[str, Any]:
        """Get supported models configuration."""
        return self.config
 
    def create_diagnostic_hooks(self, model, model_config):
        """
        Captures forward activations and backward 
        VJPs with explicit awareness of distributed tensor layouts and timing.
        """
        norm_attr = model_config["norm_attr"]
        attn_attr = model_config["attn_attr"]
        def make_forward_hook(layer_idx):
            def forward_hook(module, input, output):
                self._sync_fn()
                start_time = self._time_fn()

                tensor = output[0] if isinstance(output, tuple) else output
            
                if tensor.requires_grad and not tensor.is_leaf:
                    tensor.retain_grad()

                # Safe batch/sequence unpacking
                if tensor.dim() == 3:
                    activations = tensor[0, -1, :].clone() 
                else:
                    activations = tensor[-1, :].clone()    
                
                # Handle Distributed TP safely without destroying index mapping layout
                if _is_distributed():
                    # Avoid global reduction on local shapes unless fully gathered first.
                    # Keeping local operations ensures tracking metrics match the weight matrices slice sizes
                    # dist.all_reduce(activations, op=dist.ReduceOp.MAX)
                    warnings.warn(
                        "Distributed context detected! Local slice indices are being used. "
                        "For full Tensor Parallel (TP) support, please submit a PR implementing "
                        "dist.all_gather() synchronization for global index mappings.",
                        UserWarning
                    )

                # 1. Compute dynamic threshold
                max_val = activations.max().item()
                min_val = activations.min().item()

                positive_threshold = 0.3 * max_val if max_val > 0 else 0.0
                negative_threshold = 0.3 * min_val if min_val < 0 else 0.0

                # 2. Get masks for elements passing the threshold
                pos_active_mask = activations > positive_threshold
                neg_active_mask = activations < negative_threshold

                threshold_mask = pos_active_mask | neg_active_mask
                threshold_activations = activations[threshold_mask]

                # Initialize vector tracking tensors (0 = dead/inactive)
                top_30_vectors = torch.zeros_like(activations, dtype=torch.int)
                bottom_30_vectors = torch.zeros_like(activations, dtype=torch.int)

                # 3. Calculate cutoffs and embed native sign directions
                if threshold_activations.numel() > 0:
                    # Top 30% bucket
                    q70_cutoff = torch.quantile(threshold_activations.float(), 0.7)
                    top_mask = threshold_mask & (activations >= q70_cutoff)
                    top_30_vectors[top_mask & (activations >= 0)] = 1   # Active Positive
                    top_30_vectors[top_mask & (activations < 0)] = -1   # Active Negative

                    # Bottom 30% bucket
                    q30_cutoff = torch.quantile(threshold_activations.float(), 0.3)
                    bottom_mask = threshold_mask & (activations <= q30_cutoff)
                    bottom_30_vectors[bottom_mask & (activations >= 0)] = 1  # Active Positive
                    bottom_30_vectors[bottom_mask & (activations < 0)] = -1  # Active Negative

                # 4. Extract flat 10% sampling windows
                total_elements = activations.numel()
                active_pct = (threshold_mask.sum().item() / total_elements) * 100

                ten_pct_count = max(1, int(total_elements * 0.10))
                self._layer_metrics["ten_pct_cnt"] = ten_pct_count

                first_sample = top_30_vectors.flatten()[:ten_pct_count].tolist()
                last_sample = bottom_30_vectors.flatten()[-ten_pct_count:].tolist()

                # 5. Native Sign Color LED Renderer
                def render_vector_led(sample_list):
                    return "".join([POS_LED if p == 1 else NEG_LED if p == -1 else DEAD_LED for p in sample_list])

                def count_active(sample_list):
                    return sum(1 for p in sample_list if p != 0)

                # Structural Top-K Profiling
                top_neurons_list = []
                k_val = min(3, total_elements)
                top_val, top_idx = torch.topk(activations.abs(), k=k_val)

                for val, idx in zip(top_val.tolist(), top_idx.tolist()):
                    true_val = activations[idx].item()
                    neuron_data = {"neuron_idx": idx, "magnitude": round(val, 3), "sign": "positive" if true_val >= 0 else "negative", "source": "unknown"}
                
                    proj_norm = get_nested_attr(module, norm_attr)
                    proj_attn = get_nested_attr(module, attn_attr)

                    if proj_norm is not None:
                        neuron_data["source"] = norm_attr
                        if hasattr(proj_norm, 'weight') and proj_norm.weight is not None:
                            with torch.no_grad():
                                weight_row = proj_norm.weight.detach()
                                if idx < weight_row.size(0):
                                    scale_val = weight_row[idx].item()
                                    neuron_data["weight_scale_factor"] = round(abs(scale_val), 4)
                                    neuron_data["weight_norm"] = round(abs(scale_val), 4)

                    elif proj_attn is not None:
                        neuron_data["source"] = attn_attr
                        if hasattr(proj_attn, 'weight') and proj_attn.weight is not None:
                            with torch.no_grad():
                                weight_matrix = proj_attn.weight.detach()
            
                                if idx < weight_matrix.size(1): 
                                    weight_column = weight_matrix[:, idx]
                                    top_w_val, _ = torch.topk(weight_column.abs(), k=min(3, weight_column.size(0)))
                                    neuron_data["top_input_weights"] = [round(w, 4) for w in top_w_val.tolist()]
                                else:
                                    neuron_data["top_input_weights"] = []

                    top_neurons_list.append(neuron_data)

                # 6. Store metrics 
                if self.loop_state.step not in self._layer_metrics:
                    self._layer_metrics[self.loop_state.step] = {}
                if layer_idx in self._layer_metrics[self.loop_state.step]:
                    self._layer_metrics[self.loop_state.step][layer_idx].update({
                        "Forward_Activations": {
                            "map_first_per": render_vector_led(first_sample),
                            "map_last_per": render_vector_led(last_sample),
                            "count_first_str": f"{count_active(first_sample)}/{ten_pct_count}",
                            "count_last_str": f"{count_active(last_sample)}/{ten_pct_count}",
                            "pct": active_pct,
                            "top_neurons": json.dumps(top_neurons_list)
                        }
                    })
                else:
                    self._layer_metrics[self.loop_state.step][layer_idx] = {
                        "Forward_Activations": {
                            "map_first_per": render_vector_led(first_sample),
                            "map_last_per": render_vector_led(last_sample),
                            "count_first_str": f"{count_active(first_sample)}/{ten_pct_count}",
                            "count_last_str": f"{count_active(last_sample)}/{ten_pct_count}",
                            "pct": active_pct,
                            "top_neurons": json.dumps(top_neurons_list)
                        }
                    }

                # Nested Backward Vector Jacobian Product Hook
                def tensor_backward_hook(grad):
                    if grad is None:
                        return None
                    with torch.no_grad():
                        grads = grad[0, -1, :].clone() if grad.dim() == 3 else grad[-1, :].clone()
                
                    # Dynamic thresholding based on instantaneous grads
                    # consider upto 30% of max grads per target tensor
                    dynamic_grad_threshold = 0.3 * grads.max().item() 
                    grad_lit = (grads > dynamic_grad_threshold).int()
    
                    grad_total = len(grad_lit)
                    grad_ten_pct = max(1, int(grad_total * 0.10))
                    grad_first_sample = grad_lit[:grad_ten_pct].tolist()
                    grad_last_sample = grad_lit[-grad_ten_pct:].tolist()
                
                    if self.loop_state.step not in self._layer_metrics:
                        self._layer_metrics[self.loop_state.step] = {}
                    if layer_idx in self._layer_metrics[self.loop_state.step]:
                        self._layer_metrics[self.loop_state.step][layer_idx].update({
                            "Backward_Grads": {
                                "map_first_per": "".join([ALIVE_LED if p == 1 else DEAD_LED for p in grad_first_sample]),
                                "map_last_per": "".join([ALIVE_LED if p == 1 else DEAD_LED for p in grad_last_sample]),
                                "count_first_str": f"{sum(grad_first_sample)}/{grad_ten_pct}",
                                "count_last_str": f"{sum(grad_last_sample)}/{grad_ten_pct}",
                                "pct": (grad_lit.sum().item() / grad_total) * 100
                            }
                        })
                    else:
                        self._layer_metrics[self.loop_state.step][layer_idx] = {
                            "Backward_Grads": {
                                "map_first_per": "".join([ALIVE_LED if p == 1 else DEAD_LED for p in grad_first_sample]),
                                "map_last_per": "".join([ALIVE_LED if p == 1 else DEAD_LED for p in grad_last_sample]),
                                "count_first_str": f"{sum(grad_first_sample)}/{grad_ten_pct}",
                                "count_last_str": f"{sum(grad_last_sample)}/{grad_ten_pct}",
                                "pct": (grad_lit.sum().item() / grad_total) * 100
                            }
                        }

                    return None 

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*The .grad attribute of a Tensor that is not a leaf Tensor.*")
                    tensor.register_hook(tensor_backward_hook)
            
                self._sync_fn()
                elapsed_time_ms = (self._time_fn() - start_time) * 1000.0
                self._layer_metrics[self.loop_state.step][layer_idx]["hook_overhead_ms"] = elapsed_time_ms
            
            return forward_hook
        return make_forward_hook

    # =====================================================================
    # REGISTRATION HOOKS
    # =====================================================================

    def add_led(self, model, model_config):
        """Binds tracking infrastructure natively to execution graph sequence container."""

        
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers_container = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers_container = model.transformer.h
        else:
            raise AttributeError("Incompatible model layer architecture detected.")

        self.num_layers = len(layers_container)
        self._layer_metrics["mode"] = self.mode.value
        self._layer_metrics["num_layers"] = self.num_layers
    
        for idx in range(self.num_layers):
            layer_module = layers_container[idx]
            hook_factory = self.create_diagnostic_hooks(layer_module, model_config)
            handle = layer_module.register_forward_hook(hook_factory(idx))
            self.handles.append(handle)
        
        return 0

    def run_model(self, prompt, torch_dtype=None, context=None, model_family=None, max_new_tokens=128, verbose=True, device=None):
        """Run a prompt with full text generation and clean telemetry capture."""
        if model_family is None:
            if not self.config:
                raise ValueError("No supported models are configured.")
            model_family = next(iter(self.config))

        model_config = self.config.get(model_family)
        if model_config is None:
            raise ValueError(f"Model family '{model_family}' is not supported.")

        model_name = model_config.get("model_name")
        if not model_name:
            raise ValueError(f"'model_name' is missing for model family '{model_family}' in config.")

        if not hasattr(self, "mode"):
            self.mode = OutputType.DECISION


        print(f"Running in {self.mode} with {model_name} ...\n")
        self._layer_metrics["prompt"] = prompt
        self._layer_metrics["model"] = model_name

        torch_dtype = getattr(torch, torch_dtype.replace("torch.", "")) if isinstance(torch_dtype, str) else (torch_dtype or torch.float16)
        self._layer_metrics["torch_dtype"] = str(torch_dtype)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    
        # 1. Standardize formatting with Chat Templates
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        else:
            messages.append({"role": "system", "content": "You are a helpful, direct assistant."})

        messages.append({"role": "user", "content": prompt})

        formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask").to(device)

        # ==========================================
        # STEP 1: THE DIAGNOSTIC FIRST PASS
        # ==========================================
        # Register your hooks to catch metrics for the critical first token decision
        self.add_led(model, model_config)
        # --- Pre-Flight Memory Guard & Warning System ---
        if self.mode == OutputType.TRAJECTORY and max_new_tokens > 32:
            print(
                f"[MEMORY WARNING]: 'trajectory' mode is active with max_new_tokens={max_new_tokens}.\n"
                f"Continuous VJP backpropagation will retain deep activation graphs across {self.num_layers} layers for all predicted tokens.\n"
                f"Forward activations retained across {self.num_layers} layers for all predicted tokens.\n"
            )
    
        # ==========================================
        # STEP 2: FULL AUTOREGRESSIVE GENERATION
        # ==========================================
        # Append the first token we already calculated to our rolling sequence
    
        # Update attention mask to account for the new token
        if attention_mask is not None:
            new_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

        eos_token_id = tokenizer.eos_token_id

        initial_input_length = input_ids.shape[-1]
        for step in range(max_new_tokens):
            self.loop_state.step = step

            # 1. Manually extract embeddings from the model's embedding layer
            inputs_embeds = model.get_input_embeddings()(input_ids)
    
            # 2. Force gradient tracking on the leaf node embedding tensor
            inputs_embeds.requires_grad_(True)
    
            # 3. Forward pass (Approach 1) - Triggers your `make_forward_hook`
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
            # 4. Get logits of the last predicted token
            target_token_logits = outputs.logits[:, -1, :]
            target_token_id = torch.argmax(target_token_logits, dim=-1, keepdim=True)
            target_token_scalar = target_token_id[0, 0].item()
            target_word = tokenizer.decode([target_token_scalar])
            target_logit = outputs.logits[0, -1, target_token_scalar]
    
   
            # --- VJP / Diagnostic Trigger Point ---
            if self.mode == OutputType.DECISION:
                if step == 0:
                    target_logit.backward()
                    if device.type == 'cuda': torch.cuda.empty_cache()
                    self._layer_metrics[step]["target"] = target_word
                    for handle in self.handles:
                        handle.remove()
                    self.handles.clear()
                    self.num_tokens = 1
            else:
                target_logit.backward()
                if device.type == 'cuda': torch.cuda.empty_cache()
                self._layer_metrics[step]["target"] = target_word

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*The .grad attribute of a Tensor that is not a leaf Tensor.*")
                model.zero_grad(set_to_none=True)
                if inputs_embeds.grad is not None:
                    inputs_embeds.grad = None
            del outputs, target_token_logits, inputs_embeds

            # 5. Append the newly predicted token to the token sequence
            input_ids = torch.cat([input_ids, target_token_id], dim=-1)
            # 6. Dynamically grow the attention mask
            if attention_mask is not None:
                new_mask = torch.ones(
                    (attention_mask.shape[0], 1), device=input_ids.device, dtype=attention_mask.dtype,)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
    
            # 7. Terminate loop if EOS token is hit across the batch
            if (target_token_id == eos_token_id).all():
               self.num_tokens = step+1
               break
            if verbose:
                print(f"Generated token {step+1}...")
        if self.num_tokens == 0:
            self.num_tokens = step

        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        # Extract only the newly generated text portion
        full_response_tokens = input_ids[0, initial_input_length:]
        decoded_response = tokenizer.decode(full_response_tokens, skip_special_tokens=True)
    
        self._layer_metrics["answer"] = decoded_response.strip()
        return self._layer_metrics["answer"]

    def get_led(self) -> Dict[str, Any]:

        """
        Core Library Function: Extracts and normalizes layer metrics into a pure,
        JSON-serializable data structure. No formatting or print side-effects.
        """

        return self._layer_metrics
