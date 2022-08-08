import pandas as pd


def read(file_name):
    """
    Read a comma-separated values (csv) file into DataFrame.
    :param file_name: str
        the path and name of file.
    :return: dataframe
        A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
    """
    csv_data = pd.read_csv(file_name)
    return csv_data


def write(file_name, df):
    """
    Write object to a comma-separated values (csv) file.
    :param file_name: str
        the path and name of outputted file.
    :param df: dataframe
        the outputted dataframe.
    """
    try:
        df.to_csv(file_name, sep=",", index=False)
    except Exception as err:
        print(err)
