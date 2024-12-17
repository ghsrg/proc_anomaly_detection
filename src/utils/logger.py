import logging
from src.config.config import APP_LOG_FILE, ERROR_LOG_FILE, LOG_LEVEL, LOG_TO_SCREEN


def get_logger(name: str):
    """
    Створює логер із розділенням логів на app.log, error.log та виводом на екран.
    :param name: Назва логера.
    :return: Налаштований логер.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))  # Встановлення рівня логування

    # Формат логування
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Обробник для загальних логів (app.log)
    app_handler = logging.FileHandler(APP_LOG_FILE)
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(formatter)

    # Обробник для помилок (error.log)
    error_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # Обробник для виводу на екран з фільтрацією рівнів
    class ScreenFilter(logging.Filter):
        def filter(self, record):
            return LOG_TO_SCREEN.get(record.levelname, 0) == 1

    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.DEBUG)
    screen_handler.addFilter(ScreenFilter())
    screen_handler.setFormatter(formatter)

    # Додаємо обробники до логера
    if not logger.handlers:  # Уникаємо дублювання обробників
        logger.addHandler(app_handler)
        logger.addHandler(error_handler)
        logger.addHandler(screen_handler)

    return logger
