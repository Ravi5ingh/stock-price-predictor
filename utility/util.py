import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as sy
import pickle as pkl
import yaml as ym

def read_txt(file_name):
    """
    Read a plain text (ASCII) file
    :param file_name: The file name of the ASCII file
    :return: The text content of the ASCII file
    """

    with open(file_name, 'r') as file:
        return file.read()


def read_yaml(file_name):
    """
    Read a YAML file and return the dictionary mapping
    :param file_name: The file name of the YAML file
    :return: The dictionary mapping
    """

    with open(file_name) as file:
        return ym.load(file, Loader=ym.FullLoader)

def read_pkl(file_name):
    """
    De-serializes a pickle file into an object and returns it
    :param file_name: The name of the pickle file
    :return: The object that is de-serialized
    """

    with open(file_name, 'rb') as file:
        return pkl.load(file)

def to_pkl(obj, file_name):
    """
    Save the given object as a pickle file to the given file name
    :param obj: The object to serialize
    :param file_name: The file name to save it to
    :return: returns the same object back
    """

    with open(file_name, 'wb') as file:
        pkl.dump(obj, file)

def widen_df_display():
    """
    Widens the way dataframes are printed (setting lifetime is runtime)
    """

    pd.set_option('display.width', 3000)
    pd.set_option('display.max_columns', 100)

def to_txt(text, file_name):
    """
    Writes given string to given file as ASCII
    :param text: The string
    :param file_name: The file name
    """

    with open(file_name, "w") as text_file:
        text_file.write(text)

def update_progress(current, total, bar_length = 50):
    """
    Updates the terminal with an ASCII progress bar representing the percentage of work done
    :param current: The number of elements processed
    :param total: The total number of elements
    :param bar_length: The bar length in characters (Default: 50)
    """

    num_blocks = round(current / total * bar_length)
    done = ''.join([char * num_blocks for char in '#'])
    not_done = ''.join([char * (bar_length - num_blocks) for char in ' '])
    printover(f'[{done}{not_done}] - {current}/{total}')
    if current >= total:
        print('\n')

def printover(text):
    """
    Print over the last printed line
    :param text: The text to print
    """

    sy.stdout.write('\r' + text)

def one_hot_encode(df, column_name, prefix = '', replace_column = True, insert_to_end = False):
    """
    Performs one hot encoding on the given column in the data and replaces this column with the
    new one hot encoded columns
    :param df: The data frame in question
    :param column_name: The column to one hot encode
    :param prefix: (Optional, Default: column_name) The prefix for the new columns
    :param replace_column: (Optional, Default: True) Whether or not to replace the column to encode
    :param insert_to_end: (Optional, Default: False) Whether or not to add encoded columns at the end
    :return: The same data frame with the specified changes
    """

    dummies_insertion_index = df.columns.get_loc(column_name)
    dummies = pd.get_dummies(df[column_name], prefix=column_name if prefix == '' else prefix)

    if replace_column:
        df = df.drop([column_name], axis=1)
    else:
        dummies_insertion_index += 1

    if insert_to_end:
        df = pd.concat([df, dummies], axis=1)
    else:
        for column_to_insert in dummies.columns:
            df.insert(loc=dummies_insertion_index, column=column_to_insert, value=dummies[column_to_insert])
            dummies_insertion_index += 1

    return df

def normalize_confusion_matrix(cm_df):
    """
    Normalize the values in a confusion matrix to be between 0 and 1
    :param corr_df: The dataframe of the conusion matrix
    :return: The normalized matrix
    """

    for col in cm_df.columns:
        cm_df[col] = cm_df[col].apply(lambda x: x / cm_df[col].sum())

    return cm_df

def plot_scatter(data_frame, x, y, x_label = '', y_label = ''):
    """
    Plot a scatter plot given the data frame
    :param data_frame: The data frame to use for the scatter plot
    :param x: The column name for the x-axis
    :param y: The column name for the y-axis
    :param x_label: The label of the x-axis
    :param y_label: The label of the y-axis
    """

    x_label = x if x_label == '' else x_label
    y_label = y if y_label == '' else y_label

    data_frame = data_frame.dropna()

    standardize_plot_fonts()

    df_plot = pd.DataFrame()
    df_plot[x] = data_frame[x]
    df_plot[y] = data_frame[y]

    plot = df_plot.plot.scatter(x = x, y = y)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plot.set_title(y_label + ' vs. ' + x_label)

    plt.show()

def pad(ser, result_len, default_val = np.nan):
    """
    Pad a Series with values at the end to make it the length provided. Default padding is NaN
    :param ser: The Series
    :param result_len: The resulting length. This should be more than the current length of the series
    :param default_val: The value to pad with
    :return: The padded Series
    """

    if ser.size > result_len:
        raise ValueError('Result length ' + str(result_len) + ' needs to be more than ' + str(ser.size))

    return ser.reset_index(drop=True).reindex(range(result_len), fill_value=default_val)

def row_count(dataframe):
    """
    Gets the number of rows in a dataframe (most efficient way)
    :param dataframe: The dataframe to get the rows of
    :return: The row count
    """

    return len(dataframe.index)

def describe_hist(histogram, title, x_label, y_label):
    """
    Syntactic sugar to label the histogram axes and title
    :param histogram: The histogram
    :param title: The title to set
    :param x_label: The x-axis label to set
    :param y_label: The y-axis label to set
    """

    for ax in histogram.flatten():
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

def standardize_plot_fonts():
    """
    Standardize the title and axis fonts (Defaults to Title: 22, Axes: 15)
    """

    plt.rc('axes', labelsize=15) # Axis Font
    plt.rc('axes', titlesize=22) # Title Font

def whats(thing) :
    """
    Prints the type of object passed in
    Parameters:
        thing (Object): The object for which the type needs to be printed
    """

    print(type(thing))

def is_nan(value):
    """
    Returns true if value is NaN, false otherwise
    Parameters:
         value (Object): An object to test
    """

    return value != value

def read_csv(file_path, verbose=True):
    """
    Reads a csv file and returns the smallest possible dataframe
    :param file_path: The file path
    :param verbose: Whether or not to be verbose about the memory savings
    :return: An optimized dataframe
    """

    ret_val = pd.read_csv(file_path)
    return reduce_mem_usage(ret_val, verbose)

def reduce_mem_usage(df, verbose=True):
    """
    Takes a dataframe and returns one that takes the least memory possible.
    This works by going over each column and representing it with the smallest possible data structure.
    Example usage: my_data = pd.read_csv('D:/SomeFile.csv').pipe(reduce_mem_usage)
    Source: (https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65)
    Parameters:
        df (DataFrame): The dataframe to optimize
        verbose (bool): Whether or not to be verbose about the savings
    """

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

#region Properties

google_word2vec_model = None

word2vec_cache = None

#endregion

#region Private

def __get_confirm_token__(response):
    """
    Get a confirmation token from Google Drive (that says I'm ok with not scanning for viruses)
    :param response: The HTTP response object
    :return: The token
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def __save_response_content__(response, output_file_name):
    """
    Given an HTTP response object and a output file name, save the content to the file
    :param response: The HTTP response object
    :param output_file_name: The path of the output file
    """

    CHUNK_SIZE = 32768
    file_size = int(response.headers.get('Content-Length')) if response.headers.get('Content-Length') else None

    with open(output_file_name, "wb") as f:
        i = 1
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                mb_sofar = CHUNK_SIZE * i / 1024 / 1024
                if file_size:
                    percentage = (CHUNK_SIZE * i / file_size * 100)
                    sy.stdout.write('\r' + '[                                                  ]'
                                     .replace(' ', ':', int(percentage / 2)) + ' ' + str(
                        min(int(percentage), 100)) + '% (' + str(round(mb_sofar, 2)) + 'MB)')
                else:
                    sy.stdout.write('\r' + 'Unknown file size. ' + str(round(mb_sofar, 2)) + 'MB downloaded')
                f.write(chunk)
                i += 1
    print('')

#endregion