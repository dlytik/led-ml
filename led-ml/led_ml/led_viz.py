
from .constants import POS_LED, NEG_LED, DEAD_LED, ALIVE_LED

def gif_led(data, output_path=".", fps=1):
    """
    Independent visualization function. Generates TWO gifs simultaneously:
    1. '{model_name}_standard.gif' - Rendered with actual visible text arrows (capped at first 100).
    2. '{model_name}_extended.gif'     - High-speed pixel matrix mapping the combined 10% head and 10% tail slices.
    
    Future Work:
        - Paginated/scrollable view to explore large dimensional structures interactively without compression.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from PIL import Image
        import io
        import textwrap
    except ImportError:
        raise ImportError(
            "Visualization dependencies missing. Please install them via:\n"
            "pip install matplotlib pillow"
        )


    # Helper to convert characters to string matrices directly
    def process_raw_chars(s):
        return [char for char in s]

    # Helper to convert characters to discrete numbers for the extended pixel map
    def chars_to_numeric(s):
        mapping = {DEAD_LED: 0, ALIVE_LED: 1, POS_LED: 2, NEG_LED: 3}
        return [mapping.get(char, 0) for char in s]

    prompt = data['prompt']
    mode = data['mode']
    model_name = data['model']
    num_layers = data['num_layers']
    num_layers = int(num_layers)
    answer = data['answer']

    token_steps = sum(1 for k in data.keys() if isinstance(k, int))
    if token_steps == 0:
        token_steps = sum(1 for k in data.keys() if k.isdigit())
        is_str = True
    else:
        is_str = False

    wrapped_prompt = "\n".join(textwrap.wrap(
        f"Model: {model_name} | num of layers: {num_layers} | Mode: {mode} | Prompt: {prompt} | Answer: {answer}", 
        width=90
    ))

    # Colormaps matching your exact color scheme rules:
    # State 0: Gray (░) | State 1: Green (█) | State 2: Green (▲) | State 3: Red (▼)
    forward_cmap = ListedColormap(['#e0e0e0', '#2ca02c', '#2ca02c', '#d62728'])
    backward_cmap = ListedColormap(['#e0e0e0', '#2ca02c', '#2ca02c', '#d62728'])

    standard_frames = []
    extended_frames = []
    max_slices = 0

    # 1. Unified Frame Data Extraction Pass
    for tok in range(token_steps):
        t_key = str(tok) if is_str else tok
        target_word = data[t_key]['target']
        
        f_raw_strings = []
        b_raw_strings = []
        
        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx
            if l_key in data[t_key]:
                layer_data = data[t_key][l_key]
                
                if 'Forward_Activations' in layer_data:
                    m_f = layer_data['Forward_Activations']
                    f_str = m_f.get('map_first_per', '') + m_f.get('map_last_per', '')
                    f_raw_strings.append(f_str)
                    max_slices = max(max_slices, len(f_str))
                else:
                    f_raw_strings.append("")
                
                if 'Backward_Grads' in layer_data:
                    m_b = layer_data['Backward_Grads']
                    b_str = m_b.get('map_first_per', '') + m_b.get('map_last_per', '')
                    b_raw_strings.append(b_str)
                    max_slices = max(max_slices, len(b_str))
                else:
                    b_raw_strings.append("")
            else:
                f_raw_strings.append("")
                b_raw_strings.append("")

        if max_slices == 0:
            max_slices = 100

        # Build execution matrices for Standard Mode (Char-by-Char Text Overlay)
        f_matrix_std = [process_raw_chars((row + " " * 100)[:100]) for row in f_raw_strings]
        b_matrix_std = [process_raw_chars((row + " " * 100)[:100]) for row in b_raw_strings]

        # Build execution matrices for Full Mode (Fast Numeric Pixel Mapping)
        f_matrix_extended = [chars_to_numeric((row + " " * max_slices)[:max_slices]) for row in f_raw_strings]
        b_matrix_extended = [chars_to_numeric((row + " " * max_slices)[:max_slices]) for row in b_raw_strings]

        # --- 2. RENDER THE STANDARD FRAME (WITH VISIBLE TEXT ARROWS) ---
        fig_std, axes_std = plt.subplots(1, 2, figsize=(15, 9))
        combined_title = f"{wrapped_prompt}\n\nToken {int(tok):02d}: Token Target -> {target_word}"
        fig_std.suptitle(combined_title, fontsize=11, fontweight='bold')

        titles = ['Forward Activations Profile', 'Jacobian Causal Sensitivity (VJP)']
        std_matrices = [f_matrix_std, b_matrix_std]
        is_backward_flags = [False, True]

        for ax, matrix, title, is_backward in zip(axes_std, std_matrices, titles, is_backward_flags):
            ax.set_title(title, pad=15)
            ax.set_ylabel('Layers')
            ax.set_xlabel('Feature Slices')
            
            # Setup dimensions and grids
            ax.set_xlim(-0.5, 100 - 0.5)
            ax.set_ylim(-0.5, num_layers - 0.5)
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels(range(num_layers), fontsize=8)
            ax.set_xticks(range(0, 101, 10))

            for y_idx, row in enumerate(matrix):
                ax.axhline(y=y_idx - 0.5, color='#bfbfbf', linestyle='-', linewidth=0.5, alpha=0.5)
                for x_idx, char in enumerate(row):
                    if char != " ":
                        weight = 'bold' if char in [ALIVE_LED, POS_LED, NEG_LED] else 'normal'
                        
                        # Apply explicit color criteria for the standard text view
                        if is_backward:
                            if char == NEG_LED:
                                char_color = '#d62728'  # Red for down arrow
                            elif char in [ALIVE_LED, POS_LED]:
                                char_color = '#2ca02c'  # Green
                            else:
                                char_color = '#a0a0a0'  # Soft Gray
                        else:
                            if char in [ALIVE_LED, POS_LED]:
                                char_color = '#2ca02c'
                            elif char == NEG_LED:
                                char_color = '#d62728'
                            else:
                                char_color = '#a0a0a0'

                        ax.text(
                            x_idx, y_idx, char, 
                            ha='center', va='center', 
                            fontsize=9, fontweight=weight,
                            color=char_color,
                            fontname='DejaVu Sans'
                        )
                ax.axhline(y=num_layers - 0.5, color='#bfbfbf', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        buf_std = io.BytesIO()
        plt.savefig(buf_std, format='png', bbox_inches='tight', dpi=110)
        buf_std.seek(0)
        standard_frames.append(Image.open(buf_std))
        plt.close(fig_std)

        # --- 3. RENDER THE FULL FRAME (FAST HIGH-RES MATRIX GRAPH) ---
        fig_extended, axes_extended = plt.subplots(1, 2, figsize=(18, 9))
        fig_extended.suptitle(combined_title, fontsize=11, fontweight='bold')

        axes_extended[0].imshow(f_matrix_extended, cmap=forward_cmap, aspect='auto', origin='lower', vmin=0, vmax=3)
        axes_extended[0].set_title('Forward Activations Profile')
        axes_extended[0].set_ylabel('Layers')
        axes_extended[0].set_xlabel('Feature Slices')

        axes_extended[1].imshow(b_matrix_extended, cmap=backward_cmap, aspect='auto', origin='lower', vmin=0, vmax=3)
        axes_extended[1].set_title('Jacobian Causal Sensitivity (VJP)')
        axes_extended[1].set_xlabel('Feature Slices')

        for ax in axes_extended:
            ax.set_ylim(-0.5, num_layers - 0.5)
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels(range(num_layers), fontsize=8)
            ax.set_xticks(range(0, max_slices + 1, 50))
            ax.tick_params(axis='x', labelsize=8)
            for y in range(num_layers):
                ax.axhline(y=y - 0.5, color='#7f7f7f', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        buf_extended = io.BytesIO()
        plt.savefig(buf_extended, format='png', bbox_inches='tight', dpi=130)
        buf_extended.seek(0)
        extended_frames.append(Image.open(buf_extended))
        plt.close(fig_extended)

    model_name = model_name.strip()
    model_name = model_name.replace("/", "-")
    # 4. Save and compile out both files (Fixed Pillow Compilation syntax)
    if standard_frames:
        std_path = f"{model_name}_standard.gif"
        standard_frames[0].save(
            std_path, 
            save_all=True, 
            append_images=standard_frames[1:], 
            duration=int(1000 / fps), 
            loop=0
        )
        print(f"Successfully exported standard text-arrow map to: {std_path}")

    if extended_frames:
        extended_path = f"{model_name}_extended.gif"
        extended_frames[0].save(
            extended_path, 
            save_all=True, 
            append_images=extended_frames[1:], 
            duration=int(1000 / fps), 
            loop=0
        )
        print(f"Successfully exported complete high-res map to: {extended_path}")

def text_led(data):
    """Appends high-density execution telemetry metrics to standard output."""
    """Writes out the standard visualization table of acivations and jacobians."""
    prompt = data["prompt"]
    model_name = data["model"]
    mode = data["mode"]
    num_layers = data["num_layers"]
    cnt = data["ten_pct_cnt"]
    answer = data["answer"]

    token_steps = sum(1 for k in data.keys() if isinstance(k, int))
    if token_steps == 0:
        token_steps = sum(1 for k in data.keys() if k.isdigit())
        is_str = True
    else:
        is_str = False

    print(f"Model: {model_name} | num of layers: {num_layers} | Mode: {mode} | Prompt: {prompt}")
    print(f"Answer: {answer}")
    width = min(cnt, 60)
    print(f"{'Layer':<5} | {'Map First 10% (Forward)':<{width}} | {'Count'}\n")
    print("-" * 135 + "\n")
    for tok in range(token_steps):
        t_key = str(tok) if is_str else tok
        target_word = data[t_key]["target"]
        print(f"Step{tok:02d}   | {target_word}\n")
          
        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                layer_data = data[t_key][l_key]
            if "Forward_Activations" in layer_data:
                m = layer_data["Forward_Activations"]
                print(f"L{idx:02d}   | {m['map_first_per']:<45} | {m['count_first_str']:<7}\n")

        print(f"\n{'Layer':<5} | {'Map Last 10% (Forward)':<{cnt}} | {'Count'}\n")
        print("-" * 135 + "\n")

        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                layer_data = data[t_key][l_key]
                if "Forward_Activations" in layer_data:
                    m = layer_data["Forward_Activations"]
                    print(f"L{idx:02d}   | {m['map_last_per']:<45} | {m['count_last_str']:<7}\n")

        print(f"\n{'Layer':<5} | {'Jacobian Map First 10% (Causal sensitivity)':<{cnt}} | {'Count'}\n")
        print("-" * 135 + "\n")
        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                layer_data = data[t_key][l_key]
                if "Backward_Grads" in layer_data:
                    m = layer_data["Backward_Grads"]
                    print(f"L{idx:02d}   | {m['map_first_per']:<45} | {m['count_first_str']:<7}\n")
        print(f"\n{'Layer':<5} | {'Jacobian Map Last 10% (Causal sensitivity)':<{cnt}} | {'Count'}\n")
        print("-" * 135 + "\n")
        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                layer_data = data[t_key][l_key]
                if "Backward_Grads" in layer_data:
                    m = layer_data["Backward_Grads"]
                    print(f"L{idx:02d}   | {m['map_last_per']:<45} | {m['count_last_str']:<7}\n")
        print(f"\n{'Layer':<5} | {'Forward Density':<17} | {'Jacobian Density'}\n")
        print("-" * 60 + "\n")

        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                m = data[t_key][l_key]
                if "Forward_Activations" in m:
                   f_pct = f"{m['Forward_Activations']['pct']:.1f}%" if 'pct' in m['Forward_Activations'] else "░░░"
                if "Backward_Grads" in m:
                    b_pct = f"{m['Backward_Grads']['pct']:.1f}%" if 'pct' in m['Backward_Grads'] else "░░░"
                print(f"L{idx:02d} | {f_pct} | {b_pct}\n")

        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                m = data[t_key][l_key]
                if "hook_overhead_ms" in m:
                    elapsed_ms =  m["hook_overhead_ms"]
                    print(f"L{idx:02d} | hook overhead is {elapsed_ms:.3f} in ms\n")

        print(f"\n{'Layer':<5} {'Weights and Biases, top 3 neurons on every layer'}\n")
        print("-" * 80 + "\n")
        for idx in range(num_layers):
            l_key = str(idx) if is_str else idx 
            if l_key in data[t_key]:
                m = data[t_key][l_key]
                if "Forward_Activations" in m:
                    k = m["Forward_Activations"]
                    if 'top_neurons' in k:
                        print(f"L{idx:02d} | {k['top_neurons']}\n\n")
