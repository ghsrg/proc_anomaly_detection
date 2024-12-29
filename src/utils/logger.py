import logging
from colorama import Fore, Style, init
import xml.etree.ElementTree as ET
from src.config.config import APP_LOG_FILE, ERROR_LOG_FILE, LOG_LEVEL, LOG_TO_SCREEN
from pprint import pformat
import networkx as nx

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
        color = self.COLOR_MAP.get(record.levelno, Fore.WHITE)
        log_message = super().format(record)
        return f"{color}{log_message}{Style.RESET_ALL}"


import pandas as pd

import pandas as pd
import networkx as nx
from pprint import pformat


class CustomLogger(logging.Logger):
    def debug(self, msg, *args, **kwargs):
        if not isinstance(msg, str):
            variable_name = kwargs.pop('variable_name', '<unnamed>')
            depth = kwargs.pop('depth', None)
            max_lines = kwargs.pop('max_lines', None)

            msg_details = [
                f"Назва змінної: {variable_name}",
                f"Тип: {type(msg)}",
                f"Довжина: {len(msg) if hasattr(msg, '__len__') else 'N/A'}"
            ]

            # Рекурсивна функція для перевірки типів елементів в структурі
            def analyze_structure(data, level=3, max_depth=10):
                """
                Рекурсивно аналізує структуру даних (словники, списки, тощо).
                :param data: Дані для аналізу.
                :param level: Поточний рівень рекурсії (дефолтне значення 1).
                :param max_depth: Максимальна глибина рекурсії.
                :return: Опис структури.
                """
                if level > max_depth:
                    return "...(over max depth)..."

                if isinstance(data, dict):
                    result = {}
                    for key, value in list(data.items())[:max_lines]:
                        result[key] = analyze_structure(value, level + 1, max_depth)
                    return result
                elif isinstance(data, list):
                    return [analyze_structure(item, level + 1, max_depth) for item in data[:max_lines]]
                elif isinstance(data, pd.DataFrame):
                    # Додано: кількість рядків у DataFrame
                    return f"DataFrame - Колонки: {list(data.columns)}, Кількість рядків: {data.shape[0]}"
                elif isinstance(data, nx.Graph):
                    # Для графів NetworkX
                    return (f"Graph - Тип: {type(data).__name__}, "
                            f"Вузлів: {data.number_of_nodes()}, "
                            f"Ребер: {data.number_of_edges()}, "
                            f"Вузли (перші {max_lines}): {list(data.nodes())[:max_lines]}, "
                            f"Ребра (перші {max_lines}): {list(data.edges())[:max_lines]}")
                else:
                    return f"Тип: {type(data)}; Значення: {repr(data)}"

            # Якщо це словник, аналізуємо його структуру
            if isinstance(msg, dict):
                msg_details.append(f"Структура словника (перші {max_lines} елементів):")
                for key, value in list(msg.items())[:max_lines]:
                    msg_details.append(f"{key}: {analyze_structure(value, level=1, max_depth=3)}")
            # Якщо це DataFrame, виводимо завжди, навіть як значення в словнику
                msg_details.append("_____________\n")
            elif isinstance(msg, pd.DataFrame):
                # Виведення назв колонок
                msg_details.append(f"Колонки DataFrame: {list(msg.columns)}")
                # Виведення типів даних у кожному стовпці
                #msg_details.append(f"Типи даних у кожному стовпці: {msg.dtypes}")
                # Виведення перших кількох рядків
                msg_details.append(f"Структура DataFrame (перші {max_lines} рядків):\n{msg.head(max_lines)}")
                msg_details.append("\n_____________\n")
            # Якщо це граф NetworkX
            elif isinstance(msg, nx.Graph):
                msg_details.append(f"Граф NetworkX: {analyze_structure(msg, level=1, max_depth=3)}")
                msg_details.append("\n_____________\n")
            elif isinstance(msg, ET.Element):
                # Обробка XML Element
                children = list(msg)
                msg_details.append(f"Тип: <class 'xml.etree.ElementTree.Element'>")
                msg_details.append(f"Довжина: {len(children)}")  # Кількість дочірніх елементів
                msg_details.append("Значення:")
                #msg_details.append(f"  Тег: {msg.tag}")
                msg_details.append(f"  Атрибути: {msg.attrib}")
                if children:
                    msg_details.append(f"  Перший дочірній тег: {children[0].attrib}")
                msg_details.append("\n_____________\n")
            else:
                formatted_value = pformat(msg, indent=4, depth=depth)

                # Обрізаємо по max_lines, якщо вказано
                if max_lines:
                    formatted_value = "\n".join(formatted_value.splitlines()[:max_lines])

                msg_details.append(f"Значення:\n{formatted_value}")
                msg_details.append("\n_____________\n")
            msg = "\n".join(msg_details)


        try:
            super().debug(msg.encode('utf-8').decode('utf-8'), *args)
        except UnicodeEncodeError:
            msg = msg.encode('ascii', errors='replace').decode('ascii')
            super().debug(msg, *args)


def get_logger(name: str):
    """
    Створює логер із розділенням логів на app.log, error.log та кольоровим виводом у консоль.
    :param name: Назва логера.
    :return: Налаштований логер.
    """
    logging.setLoggerClass(CustomLogger)
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
