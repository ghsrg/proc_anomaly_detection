import argparse
from src.utils.logger import get_logger
from src.modes.experimental_mode import run_experimental_mode
from src.modes.analytical_mode import run_analytical_mode
from src.modes.production_mode import run_production_mode

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
        args = parser.parse_args()

        # Вибір режиму роботи
        if args.mode == "preparation":
            logger.info("Запущено експериментальний режим.")
            run_experimental_mode(args)
        elif args.mode == "analytical":
            logger.info("Запущено аналітичний режим.")
            run_analytical_mode(args)
        elif args.mode == "production":
            logger.info("Запущено режим виконання.")
            run_production_mode(args)
        else:
            logger.error(f"Невідомий режим: {args.mode} Допустимі значення: experimental, analytical, production.")

    except Exception as e:
        logger.critical(f"Критична помилка: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Проєкт успішно запущено.")
    main()
