from datetime import datetime
from collections import defaultdict
import pandas as pd
def date2str (value):
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%dT%H:%M:%S.%f')  # Формат ISO з мікросекундами
    return value


def convert_to_json_serializable(data):
    if isinstance(data, defaultdict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, set):
        return list(data)
    elif isinstance(data, list):
        return [convert_to_json_serializable(v) for v in data]
    elif isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    else:
        return data