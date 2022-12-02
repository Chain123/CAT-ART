import os


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

