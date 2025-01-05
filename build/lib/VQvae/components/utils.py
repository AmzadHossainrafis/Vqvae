import yaml


def read_config(config_path):
    """
    arg :
    config_path : path to the config file

    return :
    config : config file in the form of dictionary


    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config