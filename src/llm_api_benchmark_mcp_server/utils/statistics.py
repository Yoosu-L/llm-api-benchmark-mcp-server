import numpy as np
import math

def calculate_statistics(data):
    if not data:
        return {
            "total": 0.0,
            "avg": 0.0,
            "distribution": {
                "max": 0.0,
                "p50": 0.0,
                "p10": 0.0,
                "p1": 0.0,
                "min": 0.0
            }
        }
    data_np = np.array(data)
    sorted_data = np.sort(data_np)
    n = len(sorted_data)

    # Calculate p1 (actual value from sorted data)
    p1_index = min(n - 1, math.ceil(0.01 * n) - 1) if n > 0 else 0
    p1_actual = float(sorted_data[p1_index]) if n > 0 else 0.0

    # Calculate p10 (actual value from sorted data)
    p10_index = min(n - 1, math.ceil(0.10 * n) - 1) if n > 0 else 0
    p10_actual = float(sorted_data[p10_index]) if n > 0 else 0.0

    # Calculate p50 (median) - using 'lower' interpolation equivalent for actual value
    p50_index = math.floor(0.50 * (n - 1))
    p50_actual = float(sorted_data[p50_index])

    return {
        "total": float(np.sum(data_np)),
        "avg": float(np.mean(data_np)),
        "distribution": {
            "max": float(np.max(data_np)),
            "p50": p50_actual,
            "p10": p10_actual,
            "p1": p1_actual,
            "min": float(np.min(data_np))
        }
    }

def calculate_ttft_statistics(data):
    if not data:
        return {
            "avg": 0.0,
            "distribution": {
                "min": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "max": 0.0
            }
        }
    data_np = np.array(data)
    sorted_data = np.sort(data_np)
    n = len(sorted_data)

    # Calculate p50 (median) - using 'lower' interpolation equivalent for actual value
    p50_index = math.floor(0.50 * (n - 1))
    p50_actual = float(sorted_data[p50_index])

    # Calculate p90 (actual value from sorted data)
    p90_index = min(n - 1, math.ceil(0.90 * n) - 1) if n > 0 else 0
    p90_actual = float(sorted_data[p90_index]) if n > 0 else 0.0

    # Calculate p99 (actual value from sorted data)
    p99_index = min(n - 1, math.ceil(0.99 * n) - 1) if n > 0 else 0
    p99_actual = float(sorted_data[p99_index]) if n > 0 else 0.0

    return {
        "avg": float(np.mean(data_np)),
        "distribution": {
            "min": float(np.min(data_np)),
            "p50": p50_actual,
            "p90": p90_actual,
            "p99": p99_actual,
            "max": float(np.max(data_np))
        }
    }
