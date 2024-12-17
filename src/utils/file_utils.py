import os

def ensure_dir_exists(directory):
    """
    Перевіряє існування директорії, якщо її немає — створює.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
