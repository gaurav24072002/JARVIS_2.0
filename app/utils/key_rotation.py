from typing import Optional, Tuple


def get_next_key_pair(num_keys: int = 1, need_brain: bool = True) -> Tuple[Optional[int], int]:
    """Backward-compatible helper. All calls now go to a single local
    Ollama model, so brain and chat always use index 0."""
    return (0, 0)
