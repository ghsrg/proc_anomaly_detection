import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch.nn.functional import relu
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, doc_dim):
        """
        Ініціалізація згорткової нейронної мережі (CNN).

        :param input_dim: Вхідні ознаки для вузлів + ребер.
        :param hidden_dim: Розмір прихованого шару.
        :param output_dim: Кількість вихідних ознак.
        :param doc_dim: Вхідні ознаки всього документа.
        """
        super(CNN, self).__init__()
        # Шари для обробки вузлів + ребер
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Шар для обробки doc_features
        self.doc_fc = nn.Linear(doc_dim, hidden_dim)

        # Остаточний повнозв'язний шар (об'єднує локальні і глобальні ознаки)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Активації
        self.activation = nn.ReLU()  # Активація для прихованих шарів
        self.final_activation = nn.Sigmoid()  # Активація для класифікації

    def forward(self, x, doc_features=None):
        """
        Прямий прохід через CNN.

        :param x: Тензор вузлів + ребер (node features + edge features).
        :param doc_features: Ознаки документа.
        :return: Вихідний тензор після обробки.
        """
        print(f"Вхідний x: форма {x.size()}")
        print(f"Вхідний doc_features: форма {doc_features.size() if doc_features is not None else 'None'}")

        # Обробка вузлів + ребер
        x = x.unsqueeze(1).permute(0, 2, 1)  # [batch_size, channels, sequence_length]
        print(f"Після перетворення x: форма {x.size()}")
        x = self.activation(self.conv1(x))
        print(f"Після conv1 x: форма {x.size()}")
        x = self.pool(self.activation(self.conv2(x))).squeeze(2)  # [batch_size, hidden_dim]
        print(f"Після pool x: форма {x.size()}")

        # Перевірка форми перед агрегацією
        if len(x.size()) < 2:
            raise ValueError(f"Неправильна форма тензора x: {x.size()}")

        # Обробка документних ознак окремо
        if doc_features is not None:
            doc_emb = self.activation(self.doc_fc(doc_features))  # [batch_size, hidden_dim]
            print(f"Після обробки doc_features: форма {doc_emb.size()}")
        else:
            doc_emb = torch.zeros(x.size(0), self.doc_fc.out_features, device=x.device)  # [batch_size, hidden_dim]
            print(f"doc_features не задано, doc_emb: форма {doc_emb.size()}")

        # Перевірка, чи збігаються розміри
        if x.size(0) != doc_emb.size(0):
            raise ValueError(f"Розмірності batch_size не збігаються: x має {x.size(0)}, doc_emb має {doc_emb.size(0)}")

        # Об'єднання локальних і глобальних ознак
        combined = torch.cat([x, doc_emb], dim=1)  # [batch_size, hidden_dim * 2]
        print(f"Після об'єднання combined: форма {combined.size()}")

        # Остаточна класифікація
        output = self.fc(combined)  # [batch_size, output_dim]
        print(f"Вихідний output: форма {output.size()}")
        return self.final_activation(output)


def prepare_data(normal_graphs, anomalous_graphs, anomaly_type):
    """
    Підготовка даних для CNN.

    :param normal_graphs: Реєстр нормальних графів.
    :param anomalous_graphs: Реєстр аномальних графів.
    :param anomaly_type: Тип аномалії для навчання.
    :return: Дані для CNN.
    """
    data_list = []

    def transform_doc(doc_info, selected_doc_attrs):
        """
        Перетворення атрибутів документа на тензор.

        :param doc_info: Інформація про документ.
        :param selected_doc_attrs: Вибрані атрибути документа.
        :return: Тензор атрибутів документа.
        """
        return torch.tensor([doc_info.get(attr, 0.0) for attr in selected_doc_attrs], dtype=torch.float)

    for idx, row in normal_graphs.iterrows():
        # Документні атрибути
        doc_info = row.get("doc_info", {})
        selected_doc_attrs = ["PurchasingBudget", "InitialPrice", "FinalPrice"]  # Адаптуйте до ваших атрибутів
        doc_features = transform_doc(doc_info, selected_doc_attrs)

        # Вузлові атрибути
        graph = row["graph"]  # Завантаження графу
        node_features = []
        for _, node_data in graph.nodes(data=True):
            node_features.append([node_data.get("attr1", 0), node_data.get("attr2", 0)])
        node_features = torch.tensor(node_features, dtype=torch.float)

        # Реброві атрибути
        edge_features = []
        for _, _, edge_data in graph.edges(data=True):
            edge_features.append([edge_data.get("attr1", 0), edge_data.get("attr2", 0)])
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        # Додавання ребрових атрибутів як окремого виміру
        if edge_features.size(0) > 0:
            edge_features_expanded = edge_features.mean(dim=0).unsqueeze(0).expand(node_features.size(0), -1)
        else:
            edge_features_expanded = torch.zeros((node_features.size(0), 1), dtype=torch.float)

        combined_features = torch.cat([node_features, edge_features_expanded], dim=1)

        # Розширення doc_features для відповідності кількості вузлів
        doc_features_expanded = doc_features.unsqueeze(0).expand(node_features.size(0), -1)

        data = Data(
            x=combined_features,
            doc_features=doc_features,
            y=torch.tensor([0], dtype=torch.float)  # Клас (нормальний/аномальний)
        )
        data_list.append(data)

    return data_list




def train_epoch(model, data, optimizer, batch_size=24, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    model.train()
    total_loss = 0

    for batch_idx in range(0, len(data), batch_size):
        batch = data[batch_idx:batch_idx + batch_size]
        x = torch.cat([item.x for item in batch], dim=0)
        doc_features = torch.stack([item.doc_features for item in batch])
        y = torch.tensor([item.y.item() for item in batch], dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(x, doc_features=doc_features)
        loss = loss_fn(outputs, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data)

def calculate_statistics(model, data):
    model.eval()
    predictions, labels = [], []

    with torch.no_grad():
        for item in data:
            x = item.x
            doc_features = item.doc_features
            y = item.y.item()

            output = model(x, doc_features=doc_features).item()
            predictions.append(output)
            labels.append(y)

    precision, recall = 0.8, 0.7  # Замінити на фактичні розрахунки
    f1_score = 0.75

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def create_optimizer(model, learning_rate=0.001):
    return Adam(model.parameters(), lr=learning_rate)
