import argparse
from src.utils.logger import get_logger
from src.modes.experimental_mode import run_experimental_mode
from src.modes.analytical_mode import run_analytical_mode
from src.modes.production_mode import run_production_mode

logger = get_logger(__name__)  # Ініціалізація логера

def main():
    try:
        # Парсинг аргументів командного рядка
        parser = argparse.ArgumentParser(description="Аналіз аномалій у бізнес-процесах.")
        parser.add_argument("--mode", type=str, required=True,
                            help="Режим роботи: experimental, analytical, production")
        parser.add_argument("--reload", action="store_true", default=False,
                            help="Перезавантажити дані у raw (опціонально)")
        args = parser.parse_args()

        # Вибір режиму роботи
        if args.mode == "experimental":
            logger.info("Запущено експериментальний режим.")
            run_experimental_mode(args.reload)
        elif args.mode == "analytical":
            logger.info("Запущено аналітичний режим.")
            run_analytical_mode()
        elif args.mode == "production":
            logger.info("Запущено режим виконання.")
            run_production_mode()
        else:
            logger.error(f"Невідомий режим: {args.mode} Допустимі значення: experimental, analytical, production.")

    except Exception as e:
        logger.critical(f"Критична помилка: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Проєкт успішно запущено.")
    main()
