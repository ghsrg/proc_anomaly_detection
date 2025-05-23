import pandas as pd
import uuid
import json
from src.utils.logger import get_logger
from src.utils.file_utils import save_register, load_graph, save_graph, load_register
from src.utils.file_utils_l import join_path
from src.config.config import GRAPH_PATH, NORMAL_GRAPH_PATH, ANOMALOUS_GRAPH_PATH
from src.utils.graph_variations import generate_normal_graph, generate_anomalous_graph, create_doc_anomaly
from src.utils.graph_utils import clean_graph, format_graph_values, format_doc_values
from src.utils.visualizer import visualize_graph_with_dot
import traceback
from tqdm import tqdm

logger = get_logger(__name__)

def generate_variations(total_count, anomaly_type=None, min_nodes_count=50):
    """
       Генерує варіації графів (нормальних або аномальних) і фіксує їх у реєстрі.

       :param total_count: Загальна кількість графів для генерації.
       :param anomaly_type: Тип аномалії (якщо None, генеруються нормальні графи).
       :param min_nodes_count: Мінімальна кількість вузлів у графі для обробки, якщо вузлів менше, граф пропускається.
       """
    try:
        graph_register = load_register('graph_register')  # Реєстр реальних графів
        variation_register_name = 'anomalous_graphs' if anomaly_type else 'normal_graphs'
        variation_path = ANOMALOUS_GRAPH_PATH if anomaly_type else NORMAL_GRAPH_PATH
        variation_register = load_register(variation_register_name)

        # Розрахунок кількості варіацій на граф
        total_graphs = len(graph_register)
        variations_per_graph = -(-total_count // total_graphs)
        extra_variations = total_count % total_graphs  # Залишкові графи

        logger.info(f"Загальна кількість потрібних графів: {total_count}.")
        logger.info(f"Планується згенерувати {variations_per_graph} варіацій для кожного графа.")
        if extra_variations > 0:
            logger.info(f"Додаткові графи: {extra_variations}.")

        new_variations = []
        remaining_graphs = total_count  # Загальна кількість графів, яку потрібно згенерувати
        total_cycles = len(graph_register)  # Загальна кількість ітерацій

        #for current_cycle, (_, row) in enumerate(graph_register.iterrows(), start=1):
        for current_cycle, (_, row) in enumerate(
                tqdm(graph_register.iterrows(), desc="Обробка графів", total=len(graph_register)), start=1):

            if remaining_graphs <= 0:
                break  # Якщо досягли потрібної кількості, виходимо з циклу

            progress = (current_cycle / total_cycles) * 100
            print(f"Цикл {current_cycle}/{total_cycles} - Прогрес: {progress:.2f}%")

            doc_id = row['doc_id']
            root_proc_id = row['root_proc_id']
            graph_file_name = row['graph_path']
            #doc_info = row['doc_info']
            doc_info = format_doc_values(row['doc_info'], numeric_attrs=['doc_id', 'FinalPrice','InitialPrice','PurchasingBudget'], date_attrs=['DateAppCommAss', 'DateAppContract', 'DateAppFunAss', 'DateAppProcCom', 'DateApprovalFD', 'DateApprovalProcurementResults', 'DateApprovalStartProcurement', 'DateInWorkKaM', 'DateKTC', 'DateSentSO', 'ExpectedDate', 'doc_createdate'], default_numeric=0, default_date='2000-01-01T00:00:00')

            try:
                # Завантаження оригінального графа
                orig_graph = load_graph(file_name=graph_file_name, path=GRAPH_PATH)
                #cl_graph = clean_graph(orig_graph)
                cl_graph = orig_graph
                if cl_graph.number_of_nodes() < 5: # Пропускаємо графи з малою кількістю вузлів, які точно не релевантні
                    continue
                graph = format_graph_values(cl_graph, numeric_attrs=['active_executions', 'DURION_', 'DURATION_E', 'SEQUENCE_COUNTER_', 'PurchasingBudget', 'InitialPrice', 'FinalPrice', 'duration_work', 'duration_work_E'], date_attrs=['doc_createdate', 'DateSentSO', 'DateAppContract', 'DateAppProcCom', 'DateApprovalProcurementResults', 'DateAppCommAss', 'DateAppFunAss', 'DateApprovalStartProcurement', 'DateApprovalFD', 'DateInWorkKaM', 'DateKTC', 'ExpectedDate', 'END_TIME_', 'START_TIME_', 'first_view'], default_numeric=0, default_date='2000-01-01T00:00:00.0' )
                del orig_graph
                #del cl_graph
                # Розрахунок кількості варіацій для поточного графа
                variations_for_this_graph = variations_per_graph
                if current_cycle <= extra_variations:  # Додаткові варіації для перших графів
                    variations_for_this_graph += 1

                # Перша варіація — оригінал або аномалія
                original_id = str(uuid.uuid4())
                if anomaly_type:
                    anomalous_graph, params = generate_anomalous_graph(graph, anomaly_type=anomaly_type)
                    save_graph(anomalous_graph, f"{original_id}_{graph_file_name}", variation_path)
                    del anomalous_graph
                    #visualize_graph_with_dot(anomalous_graph, join_path([variation_path, f"{original_id}_{graph_file_name}"]))
                    doc_info = create_doc_anomaly (doc_info, anomaly_type)
                    new_variations.append({
                        'id': original_id,
                        'doc_id': doc_id,
                        'root_proc_id': root_proc_id,
                        'graph_path': f"{original_id}_{graph_file_name}",
                        'date': pd.Timestamp.now().date(),
                        'params': json.dumps(params),  # Збереження параметрів як JSON-рядок
                        'doc_info': doc_info  # Збереження параметрів документа
                    })
                    logger.info(f"Збережено оригінальний аномальний граф {graph_file_name} з типом аномалії {anomaly_type}.")
                else:
                    save_graph(graph, f"{original_id}_{graph_file_name}", variation_path)
                    #visualize_graph_with_dot(graph, join_path([variation_path, f"{original_id}_{graph_file_name}"]))
                    #del graph
                    new_variations.append({
                        'id': original_id,
                        'doc_id': doc_id,
                        'root_proc_id': root_proc_id,
                        'graph_path': f"{original_id}_{graph_file_name}",
                        'date': pd.Timestamp.now().date(),
                        'params': json.dumps({'type': 'original'}),  # Збереження параметрів як JSON-рядок
                        'doc_info': doc_info  # Збереження параметрів документа
                    })
                    logger.info(f"Збережено оригінальний граф {graph_file_name}.")
                if cl_graph.number_of_nodes() < min_nodes_count:    #   Пропускаємо графи з малою кількістю вузлів, які не бажані для генерації варіацій
                    continue
                # Наступні варіації — модифікації або аномалії
                for _ in range(variations_for_this_graph - 1):  # -1, бо оригінал уже враховано
                    new_id = str(uuid.uuid4())
                    if anomaly_type:
                        generated_graph, params = generate_anomalous_graph(graph, anomaly_type=anomaly_type)
                        doc_info = create_doc_anomaly(doc_info, anomaly_type)
                    else:
                        generated_graph, params = generate_normal_graph(graph)

                    file_name = f"{new_id}_{graph_file_name}"
                    save_graph(generated_graph, file_name, variation_path)
                    del generated_graph
                    #if anomaly_type:
                        #visualize_graph_with_dot(generated_graph, join_path([variation_path, file_name]))

                    new_variations.append({
                        'id': new_id,
                        'doc_id': doc_id,
                        'root_proc_id': root_proc_id,
                        'graph_path': file_name,
                        'date': pd.Timestamp.now().date(),
                        'params': json.dumps(params),  # Збереження параметрів як JSON-рядок
                        'doc_info': doc_info  # Збереження параметрів документа
                    })
                    logger.info(f"Згенеровано варіацію графа {file_name}.")

                # Зменшення залишкової кількості графів
                remaining_graphs -= variations_for_this_graph

            except Exception as e:
                logger.error(f"Помилка під час обробки графа {graph_file_name}: {e}")
                logger.error(f"Деталі помилки:\n{traceback.format_exc()}")

        # Оновлення реєстру
        if new_variations:
            variation_register = pd.concat([variation_register, pd.DataFrame(new_variations)], ignore_index=True)
            save_register(variation_register, variation_register_name)
            logger.info(f"Додано {len(new_variations)} варіацій до реєстру {variation_register_name}.")

    except Exception as e:
        logger.critical(f"Критична помилка у функції generate_variations: {e}")
        logger.error(f"Деталі помилки:\n{traceback.format_exc()}")

