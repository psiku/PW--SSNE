MAP = {0: 'bach', 1: 'beethoven', 2: 'debussy', 3: 'scarlatti', 4: 'victoria'}
REVERSE_MAP = {v: k for k, v in MAP.items()}


def map_label_to_name(label: int) -> str:
    return MAP.get(label, 'unknown')

def map_name_to_label(name: str) -> int:
    return REVERSE_MAP.get(name, -1)