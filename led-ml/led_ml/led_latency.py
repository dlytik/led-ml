import sys
from pathlib import Path
import statistics

def extract_layer_overhead(data):
    """
    Extract hook_overhead_ms for each layer across all tokens.
    
    Structure: data["token_id"]["layer_id"]["hook_overhead_ms"]
    We collect all hook_overhead_ms values for each layer across all tokens.
    
    Returns:
        dict: {layer_name: [overhead_values]}, where overhead_values span all tokens
    """
    layer_data = {}
    
    # Iterate through tokens (outer numeric keys: "0", "1", "2", etc.)
    for token_key, token_value in data.items():
        # Skip non-token keys (metadata like 'prompt', 'model', 'mode', etc.)
        if not str(token_key).isdigit():
            continue
        
        if not isinstance(token_value, dict):
            continue
        
        # Inside each token, iterate through layers (keys: "0", "1", ..., ")
        for layer_key, layer_value in token_value.items():
            if not str(layer_key).isdigit():
                continue
            
            if not isinstance(layer_value, dict):
                continue
            
            # Extract hook_overhead_ms for this layer in this token
            if 'hook_overhead_ms' in layer_value:
                if layer_key not in layer_data:
                    layer_data[layer_key] = []
                layer_data[layer_key].append(layer_value['hook_overhead_ms'])
    
    return layer_data

def compute_statistics(layer_data):
    """
    Compute mean and standard deviation for each layer.
    
    Args:
        layer_data: {layer_name: [overhead_values]}
    
    Returns:
        list of tuples: [(layer, mean, std, count), ...]
    """
    stats = []
    
    # Sort layers numerically
    sorted_layers = sorted(layer_data.keys(), key=lambda x: int(x))
    
    for layer in sorted_layers:
        values = layer_data[layer]
        if len(values) > 0:
            mean_val = statistics.mean(values)
            if len(values) > 1:
                std_val = statistics.stdev(values)
            else:
                std_val = 0.0
            count = len(values)
            stats.append((layer, mean_val, std_val, count))
    
    return stats

def print_table(headers, data):
    """Print a formatted ASCII table."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(separator)
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    print(header_row)
    print(separator)
    
    # Print data rows
    for row in data:
        data_row = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        print(data_row)
    
    print(separator)

def lat_led(layer_metrics, verbose=True):
    
    # Extract layer overhead data
    layer_data = extract_layer_overhead(layer_metrics)
    
    if not layer_data:
        print("Error: No layer data found in layer metrics", file=sys.stderr)
        return None
    
    # Compute statistics
    stats = compute_statistics(layer_data)
    
    # Create table
    headers = ['Layer', 'Mean (ms)', 'Std Dev (ms)', 'Token Count']
    
    # Format data for display
    table_data = []
    for layer, mean_val, std_val, count in stats:
        n = int(layer) + 1
        table_data.append([
            f"layer{n}",
            f"{mean_val:.6f}",
            f"{std_val:.6f}",
            str(count)
        ])

    if verbose: 
        # Print table with manual formatting
        print()
        print_table(headers, table_data)
        print()
    
        # Summary statistics
        all_means = [m for _, m, _, _ in stats]
        all_stds = [s for _, _, s, _ in stats]
    
        print(f"Summary Statistics Across All Layers:")
        print(f"  Overall Mean of Means: {statistics.mean(all_means):.6f} ms")
        print(f"  Overall Std Dev of Means: {statistics.stdev(all_means) if len(all_means) > 1 else 0:.6f} ms")
        print(f"  Min Layer Mean: {min(all_means):.6f} ms (layer{stats[all_means.index(min(all_means))][0]})")
        print(f"  Max Layer Mean: {max(all_means):.6f} ms (layer{stats[all_means.index(max(all_means))][0]})")
        print()

    return table_data
