import argparse
from src.utils.logger import get_logger
from src.modes.preparation_mode import run_preparation_mode
from src.modes.learning_mode import run_learning_mode
from src.modes.analityc_learn_mode import run_analitics_learn_mode
from src.modes.analityc_test_mode import run_analitics_test_mode
from src.modes.testing_mode import run_testing_mode
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

def main():
    try:
        # Парсинг аргументів командного рядка
        parser = argparse.ArgumentParser(description="Аналіз аномалій у бізнес-процесах.")
        parser.add_argument("--mode", type=str, required=True,
                            help="Режим роботи: preparation , learning, analityc, testing")
        parser.add_argument("--raw_reload", action="store_true", default=False,
                            help="Перезавантажити дані у raw (опціонально)")
        parser.add_argument("--doc2graph", action="store_true", default=False,
                            help="Проаналізувати документи й згенерувати первинні графи (опціонально)")
        parser.add_argument("--graph_synthesis", action="store_true", default=False,
                            help="Генерація додаткових графів для навчання (опціонально)")
        parser.add_argument("--normalize", action="store_true", default=False,
                            help="Запуск нормальізації графів перед режимом навчання (опціонально)")
        parser.add_argument("--model_type", type=str,
                            help="Модель навчання: model_type: GNN, CNN, RNN....")
        parser.add_argument("--anomaly_type", type=str,
                            help="Тип аномалії: missing_steps, duplicate_steps, wrong_route, abnormal_duration....")
        parser.add_argument("--action", type=str,
                            help="Тип запуску: start / resume / retrain")
        parser.add_argument("--checkpoint", type=str,
                            help="Назва checkpoint для віднрвлення навчання")
        parser.add_argument("--normal_var", type=int,
                            help="Кількість згенерованих нормальних графів ")
        parser.add_argument("--quant", type=int,
                            help="Кількість згенерованих аномальних графів duplicate_steps")
        parser.add_argument("--data_file", type=str,
                            help="Посилання на файл з підготовленими даними ")
        parser.add_argument("--pr_mode", type=str,
                            help="Тип prediction: bpmn or log ")
        parser.add_argument("--eval_only", action="store_true",
                            help="Лише інференс/оцінка без тренування")
        args = parser.parse_args()

        # Вибір режиму роботи
        if args.mode == "preparation":
            logger.info("Запущено режим підготовки даних")
            run_preparation_mode(args)
        elif args.mode == "learning":
            logger.info("Запущено режим навчання")
            run_learning_mode(args)
        elif args.mode == "analityc_learn":
            logger.info("Запущено режим аналітики навчання.")
            run_analitics_learn_mode(args)
        elif args.mode == "analityc_test":
            logger.info("Запущено режим аналітики тестування.")
            run_analitics_test_mode(args)
        elif args.mode == "testing":
            logger.info("Запущено режим тестування залежності від довжини префікса")
            run_testing_mode(args)
        else:
            logger.error(f"Невідомий режим: {args.mode} Допустимі значення: experimental, analytical, production.")

    except Exception as e:
        logger.critical(f"Критична помилка: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Проєкт успішно запущено.")
    main()
