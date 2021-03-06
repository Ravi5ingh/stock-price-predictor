import utility.util as ut
import os

def is_app_debug():
    """
    Should the flask app run in debug mode?
    :return: A boolean
    """

    return ut.read_yaml(__yaml_config_file_name__)['debug']

def get_training_cutoff_date():
    """
    Get the training cut-off date
    :return: A Date
    """

    return ut.read_yaml(__yaml_config_file_name__)['training_date_cutoff']

#region Private

__path_delimiter_mapping__ = {
    'nt': '\\',
    'posix': '/'
}

__path_delimiter__ = __path_delimiter_mapping__[os.name]

__yaml_config_file_name__ = __file__[0:__file__.rindex(__path_delimiter__)] + f'{__path_delimiter__}config.yaml'

#endregion