import yaml


def load_config(ymlfile) -> dict():
    # Load configurations
    file = open(ymlfile, "r")
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return cfg
