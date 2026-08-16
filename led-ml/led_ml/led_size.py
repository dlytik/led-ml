import sys

def ram_led(obj, seen=None):
    """Recursively calculates the true RAM footprint of nested Python objects."""
    if obj is None:
        return 0
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    # Get base size of the current object container
    size = sys.getsizeof(obj)
    
    # Recursively add inner contents
    if isinstance(obj, dict):
        size += sum(ram_led(k, seen) + ram_led(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(ram_led(item, seen) for item in obj)
    return size
