# coding=utf-8

import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from src.configuration_file import configuration_file

path = configuration_file().rootpath


def write_excel(excel_path, data):
    """
    write result into excel
    :param excel_path:
    :param data:
    :return:
    """
    dir_name = str(os.path.split(excel_path)[0])
    mkdir(dir_name)
    wb = Workbook()
    ws = wb.active
    for _ in data:
        ws.append(_)
    wb.save(excel_path)


def mkdir(path):
    """
    make directory
    :param path:
    :return:
    """
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def transform_data(data_y):
    """
    Modifying the number of defect which is greater than 1 into 1
    :param data_y: the number of defect
    :return: label: 0 or -1
    """
    y_list = []
    for i in data_y:
        if int(i) >= 1:
            y_list.append(1)
        else:
            y_list.append(-1)
    return y_list


def spilt_data(original_data):
    """
    Spliting features and labels
    :param original_data: origianl training or testing datasets
    :return: features, labels
    """
    original_data = original_data.iloc[:, :]
    original_data = np.array(original_data)

    k = len(original_data[0])

    original_data_X = original_data[:, 0:k - 1]
    original_data_y = list(original_data[:, k - 1])

    original_data_X = pd.DataFrame(original_data_X)
    original_data_y = transform_data(original_data_y)
    original_data_y = pd.DataFrame(original_data_y)

    return original_data_X, original_data_y
