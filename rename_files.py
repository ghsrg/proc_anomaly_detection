import os
import re

# Шлях до папки з файлами
folder = "C:/SerG/Proj/Python/scientific/proc_anomaly_detection/data/outputs/test_diagrams/graph"

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    # пропускаємо якщо це не файл
    if not os.path.isfile(old_path):
        continue

    # Приклад: DeepGCN_pr_logs_seed9467_pref-len_top5
    # 1) Вирізаємо seedXXXX і pref-len
    new_name = re.sub(r'_seed\d+', '', filename)       # прибирає _seed9467
    new_name = new_name.replace('_pref-len', '')       # прибирає _pref-len

    # 2) Замінюємо _pr_logs -> _logs, _pr_bpmn -> _bpmn
    new_name = new_name.replace('_pr_logs', '_logs')
    new_name = new_name.replace('_pr_bpmn', '_bpmn')

    # Формуємо новий шлях
    new_path = os.path.join(folder, new_name)

    # Перейменовуємо
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_name}")
