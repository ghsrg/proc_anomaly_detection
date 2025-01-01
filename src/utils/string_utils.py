from datetime import datetime
import pandas as pd
def date2str (value):
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%dT%H:%M:%S.%f')  # Формат ISO з мікросекундами
    return value