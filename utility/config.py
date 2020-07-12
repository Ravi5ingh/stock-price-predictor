import utility.util as ut

def is_app_debug():
    """
    Should the flask app run in debug mode?
    :return: A boolean
    """

    return ut.read_yaml(__yaml_config_file_name__)['debug']

#region Private

__yaml_config_file_name__ = __file__[0:__file__.rindex('\\')] + '\\config.yaml'

#endregion