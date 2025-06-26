import re

def parse_concurrency_levels(concurrency_str: str) -> list[int]:
    """
    Parses a comma-separated string of concurrency levels.
    """
    str_levels = concurrency_str.split(',')
    concurrency_levels = []
    for level_str in str_levels:
        level_str = level_str.strip()
        try:
            level = int(level_str)
        except ValueError:
            raise ValueError(f"Invalid concurrency level: {level_str}")
        if level <= 0:
            raise ValueError(f"Concurrency level must be positive: {level}")
        concurrency_levels.append(level)
    
    concurrency_levels.sort()
    return concurrency_levels