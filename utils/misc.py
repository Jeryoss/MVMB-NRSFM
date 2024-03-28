import os


def ensuredir(path):
    """
    Creates a folder if it doesn't exists.

    :param path: path to the folder to create
    """
    if len(path) == 0:
        return
    if not os.path.exists(path):
        os.makedirs(path)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
