import configparser


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance


@singleton
class SudokuConfig:
    def __init__(self):
        self.Config = configparser.ConfigParser()
        data_set = self.Config.read('../params.ini')
        # If none of the named files exist, the ConfigParser instance will contain an empty dataset.
        if not data_set:
            raise ValueError("Failed to open/find all config files")

        print(self.Config.sections())
        print('Loading config')
