import argparse
from src.utils.logger import get_logger
from src.modes.preparation_mode import run_experimental_mode
from src.modes.learning_mode import run_learning_mode
from src.modes.analityc_mode import run_production_mode

logger = get_logger(__name__)

def main():
    try:
        # Парсинг аргументів командного рядка
        parser = argparse.ArgumentParser(description="Аналіз аномалій у бізнес-процесах.")
        parser.add_argument("--mode", type=str, required=True,
                            help="Режим роботи: preparation , analytical, production")
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
        parser.add_argument("--missing_steps", type=int,
                            help="Кількість згенерованих аномальних графів missing_steps")
        parser.add_argument("--duplicate_steps", type=int,
                            help="Кількість згенерованих аномальних графів duplicate_steps")
        parser.add_argument("--data_file", type=str,
                            help="Посилання на файл з підготовленими даними ")
        args = parser.parse_args()

        # Вибір режиму роботи
        if args.mode == "preparation":
            logger.info("Запущено режим підготовки даних")
            run_experimental_mode(args)
        elif args.mode == "learning":
            logger.info("Запущено режим навчання")
            run_learning_mode(args)
        elif args.mode == "analityc":
            logger.info("Запущено режим аналітики.")
            run_production_mode(args)
        else:
            logger.error(f"Невідомий режим: {args.mode} Допустимі значення: experimental, analytical, production.")

    except Exception as e:
        logger.critical(f"Критична помилка: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Проєкт успішно запущено.")
    main()
