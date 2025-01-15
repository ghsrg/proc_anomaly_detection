import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.config.config import REPORTS_PATH
from src.utils.file_utils_l import join_path

# Example data creation similar to the given structure
data = {
    "Type": ["missing_steps"] * 3 + ["duplicate_steps"] * 3 + ["wrong_route"] * 3 + ["abnormal_duration"] * 3,
    "Metric": ["MAX", "AVG", "MIN"] * 4,
    "GNN": [0.829199789, 0.814285714, 0.798559254, 0.836552503, 0.826704545, 0.819659189,
            0.828860645, 0.814345992, 0.809772319, 0.81021139, 0.798793103, 0.785974685],
    "CNN": [0.800825729, 0.767716535, 0.731247113, 0.816231295, 0.805327869, 0.787018155,
            0.788898024, 0.779591837, 0.738178711, 0.799930165, 0.783464567, 0.752602118],
    "RNN": [0.839133795, 0.790933063, 0.763855624, 0.786836793, 0.780684105, 0.761559228,
            0.831047848, 0.800403226, 0.778199988, 0.811876947, 0.805084746, 0.790356824],
    "Transformers": [0.812681229, 0.804928131, 0.771027697, 0.804702765, 0.791932059, 0.787391154,
                     0.853118214, 0.844897959, 0.831202794, 0.823863716, 0.816, 0.805582466],
    "Autoencoder": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)
metric= 'Precision'
# Plotting for each type of anomaly
types = df['Type'].unique()
for anomaly_type in types:
    subset = df[df['Type'] == anomaly_type]
    x_labels = ['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder']
    metrics = ['MAX', 'AVG', 'MIN']

    values = {metric: subset[subset['Metric'] == metric][x_labels].values.flatten() for metric in metrics}

    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, values['AVG'], marker='o', label='Average', color='blue')
    plt.fill_between(x_labels, values['MIN'], values['MAX'], color='gray', alpha=0.3, label='Min-Max Range')

    plt.title(f'{metric} for {anomaly_type}')
    plt.xlabel('Model Architecture')
    plt.ylabel(metric)
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    file_name = join_path([REPORTS_PATH, f"Diagram_{metric}_{anomaly_type}.png"])
    print(file_name)
    # Спроба зберегти графік
    try:
        plt.savefig(file_name, dpi=100)
        print(f"Збережено: {file_name}")
    except Exception as e:
        print(f"Помилка при збереженні файлу: {e}")

    #plt.show()

# Example data creation similar to the given structure
data = {
    "Type": ["missing_steps"] * 3 + ["duplicate_steps"] * 3 + ["wrong_route"] * 3 + ["abnormal_duration"] * 3,
    "Metric": ["MAX", "AVG", "MIN"] * 4,
    "GNN": [1, 1, 0.99818112, 0.747203136, 0.739795918, 0.720431575, 1, 1, 1, 0.920168428, 0.901477833, 0.896498193],
    "CNN": [1, 1, 0.98991212, 1, 1, 1, 0.998966517, 0.997461929, 0.984875536, 0.999137159, 0.997487437, 0.986186274],
    "RNN": [1, 1, 0.9988115, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Transformers": [0.968642594, 0.966836735, 0.957225929, 1, 1, 1, 0.975278132, 0.961352657, 0.953257623, 1, 0.995098039, 0.989356144],
    "Autoencoder": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

}

df = pd.DataFrame(data)
metric= 'Recall'
# Plotting for each type of anomaly
types = df['Type'].unique()
for anomaly_type in types:
    subset = df[df['Type'] == anomaly_type]
    x_labels = ['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder']
    metrics = ['MAX', 'AVG', 'MIN']

    values = {metric: subset[subset['Metric'] == metric][x_labels].values.flatten() for metric in metrics}

    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, values['AVG'], marker='o', label='Average', color='blue')
    plt.fill_between(x_labels, values['MIN'], values['MAX'], color='gray', alpha=0.3, label='Min-Max Range')

    plt.title(f'{metric} for {anomaly_type}')
    plt.xlabel('Model Architecture')
    plt.ylabel(metric)
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    file_name = join_path([REPORTS_PATH, f"Diagram_{metric}_{anomaly_type}.png"])
    print(file_name)
    # Спроба зберегти графік
    try:
        plt.savefig(file_name, dpi=100)
        print(f"Збережено: {file_name}")
    except Exception as e:
        print(f"Помилка при збереженні файлу: {e}")

    #plt.show


# Example data creation similar to the given structure
data = {
    "Type": ["missing_steps"] * 3 + ["duplicate_steps"] * 3 + ["wrong_route"] * 3 + ["abnormal_duration"] * 3,
    "Metric": ["MAX", "AVG", "MIN"] * 4,
    "GNN": [1, 1, 0.99818112, 0.747203136, 0.739795918, 0.720431575, 1, 1, 1, 0.920168428, 0.901477833, 0.896498193],
    "CNN": [1, 1, 0.98991212, 1, 1, 1, 0.998966517, 0.997461929, 0.984875536, 0.999137159, 0.997487437, 0.986186274],
    "RNN": [1, 1, 0.9988115, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Transformers": [0.968642594, 0.966836735, 0.957225929, 1, 1, 1, 0.975278132, 0.961352657, 0.953257623, 1, 0.995098039, 0.989356144],
    "Autoencoder": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

}

df = pd.DataFrame(data)
metric= 'ROC AUC'
# Plotting for each type of anomaly
types = df['Type'].unique()
for anomaly_type in types:
    subset = df[df['Type'] == anomaly_type]
    x_labels = ['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder']
    metrics = ['MAX', 'AVG', 'MIN']

    values = {metric: subset[subset['Metric'] == metric][x_labels].values.flatten() for metric in metrics}

    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, values['AVG'], marker='o', label='Average', color='blue')
    plt.fill_between(x_labels, values['MIN'], values['MAX'], color='gray', alpha=0.3, label='Min-Max Range')

    plt.title(f'{metric} for {anomaly_type}')
    plt.xlabel('Model Architecture')
    plt.ylabel(metric)
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    file_name = join_path([REPORTS_PATH, f"Diagram_{metric}_{anomaly_type}.png"])
    print(file_name)
    # Спроба зберегти графік
    try:
        plt.savefig(file_name, dpi=100)
        print(f"Збережено: {file_name}")
    except Exception as e:
        print(f"Помилка при збереженні файлу: {e}")

    #plt.show()

# Example data creation similar to the given structure
data = {
    "Type": ["missing_steps"] * 3 + ["duplicate_steps"] * 3 + ["wrong_route"] * 3 + ["abnormal_duration"] * 3,
    "Metric": ["MAX", "AVG", "MIN"] * 4,
    "GNN": [0.906625721, 0.879332624, 0.87728097, 0.789357446, 0.781671159, 0.766848778, 0.906422966, 0.897674419, 0.894888612, 0.861696297, 0.84137931, 0.837606233],
    "CNN": [0.889398364, 0.868596882, 0.841142837, 0.898818666, 0.891402715, 0.88081719, 0.881591075, 0.876146789, 0.84386682, 0.888504773, 0.888888889, 0.853704662],
    "RNN": [0.912531538, 0.892018779, 0.865674263, 0.893703594, 0.891774892, 0.884642206, 0.907729253, 0.899159664, 0.875267116, 0.903172279, 0.8938317757009, 0.878904249283],
    "Transformers": [0.88583442, 0.883928571, 0.874096519, 0.9198426586, 0.917808219, 0.909050745, 0.910117262, 0.902777778, 0.888056011, 0.903426839, 0.893831776, 0.882061528],
    "Autoencoder": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)
metric= 'F1-Score'
# Plotting for each type of anomaly
types = df['Type'].unique()
for anomaly_type in types:
    subset = df[df['Type'] == anomaly_type]
    x_labels = ['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder']
    metrics = ['MAX', 'AVG', 'MIN']

    values = {metric: subset[subset['Metric'] == metric][x_labels].values.flatten() for metric in metrics}

    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, values['AVG'], marker='o', label='Average', color='blue')
    plt.fill_between(x_labels, values['MIN'], values['MAX'], color='gray', alpha=0.3, label='Min-Max Range')

    plt.title(f'{metric} for {anomaly_type}')
    plt.xlabel('Model Architecture')
    plt.ylabel(metric)
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    file_name = join_path([REPORTS_PATH, f"Diagram_{metric}_{anomaly_type}.png"])
    print(file_name)
    # Спроба зберегти графік
    try:
        plt.savefig(file_name, dpi=100)
        print(f"Збережено: {file_name}")
    except Exception as e:
        print(f"Помилка при збереженні файлу: {e}")

    #plt.show()
# Example data
metric = 'Time per epoch'
# Середні значення часу для кожної архітектури
architectures = ['GNN', 'CNN', 'RNN', 'Transformers', 'Autoencoder']
average_times = [32.6, 13.5, 282.3, 1575.3, 0]  # Значення для кожної архітектури

# Створення графіка
fig, ax = plt.subplots(figsize=(8, 6))

bar_width = 0.5  # Ширина стовпчиків
index = np.arange(len(architectures))  # Індекси для кожної архітектури

# Створення стовпчиків
ax.bar(index, average_times, width=bar_width, color='skyblue', edgecolor='black')

# Додавання підписів
ax.set_xlabel('Architecture')
ax.set_ylabel('Average Time (Seconds)')
ax.set_title('Average Time per Epoch for Each Architecture')
ax.set_xticks(index)
ax.set_xticklabels(architectures)

# Додавання значень над стовпчиками
for i, v in enumerate(average_times):
    ax.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=10)

# Show the plot
plt.tight_layout()
file_name = join_path([REPORTS_PATH, f"Diagram_{metric}.png"])
print(file_name)
# Спроба зберегти графік
try:
    plt.savefig(file_name, dpi=100)
    print(f"Збережено: {file_name}")
except Exception as e:
    print(f"Помилка при збереженні файлу: {e}")