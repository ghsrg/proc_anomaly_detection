import os
import csv

# 📂 Вкажи тут папку з графіками
folder = "C:/SerG/Proj/Python/scientific/proc_anomaly_detection/data/outputs/test_diagrams/graph"

# 📝 CSV-файл для збереження індексу
csv_file = os.path.join(folder, "plots_index.csv")

# Поля в CSV
fields = ["model", "mode", "metric", "filename"]

rows = []

for filename in os.listdir(folder):
    if not filename.endswith(".png"):
        continue

    # Розбираємо назву: Model_mode_metric.png
    base = filename.replace(".png", "")
    parts = base.split("_")

    if len(parts) < 3:
        continue

    model = parts[0]                  # APPNP, DeepGCN, GATConv, ...
    mode = parts[1]                   # bpmn або logs
    metric = "_".join(parts[2:])      # accuracy, f1, conf, top3, top5, out_of_scope

    rows.append([model, mode, metric, filename])

# Запис у CSV
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(rows)

print(f"CSV index saved to: {csv_file}")
