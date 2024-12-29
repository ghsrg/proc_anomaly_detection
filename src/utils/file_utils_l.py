import os
def join_path(path_components):
    return os.path.join(*path_components)  # Поєднує компоненти у правильний шлях

def make_dir(path):
    """
    Перевіряє існування директорії, якщо її немає — створює.
    """
    os.makedirs(path, exist_ok=True)  # Створює директорію, якщо її ще немає


def is_file_exist(path):
    return os.path.exists(path)

