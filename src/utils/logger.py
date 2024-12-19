import logging
from colorama import Fore, Style, init
from src.config.config import APP_LOG_FILE, ERROR_LOG_FILE, LOG_LEVEL, LOG_TO_SCREEN

# Ініціалізація colorama
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """
    Форматер для кольорового виводу залежно від рівня логування.
    """
    COLOR_MAP = {
        logging.DEBUG: Fore.WHITE,
        logging.INFO: Fore.LIGHTWHITE_EX,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.LIGHTRED_EX + Style.BRIGHT
    }

    def format(self, record):
        # Додаємо колір до всієї строки
        color = self.COLOR_MAP.get(record.levelno, Fore.WHITE)
        log_message = super().format(record)
        return f"{color}{log_message}{Style.RESET_ALL}"


def get_logger(name: str):
    """
    Створює логер із розділенням логів на app.log, error.log та кольоровим виводом у консоль.
    :param name: Назва логера.
    :return: Налаштований логер.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))  # Встановлення рівня логування

    # Формат для файлів
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Формат для консолі
    console_formatter = ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Обробник для app.log
    app_handler = logging.FileHandler(APP_LOG_FILE)
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(file_formatter)

    # Обробник для error.log
    error_handler = logging.FileHandler(ERROR_LOG_FILE)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Обробник для консолі
    class ScreenFilter(logging.Filter):
        def filter(self, record):
            return LOG_TO_SCREEN.get(record.levelname, 0) == 1

    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.DEBUG)
    screen_handler.addFilter(ScreenFilter())
    screen_handler.setFormatter(console_formatter)

    # Додаємо обробники
    if not logger.handlers:
        logger.addHandler(app_handler)
        logger.addHandler(error_handler)
        logger.addHandler(screen_handler)

    return logger
