# coding=utf-8

import warnings

from src.PerformanceMeasure import PerformanceMeasure
from src.classification import *

warnings.filterwarnings('ignore')


def Pofb(preds, testingcode, testing_data_value):
    """
    calculate PofB
    :param preds:
    :param testingcode:
    :param testing_data_value:
    :return:
    """
    pred = [p[1] for p in preds]

    # calculate defect density
    defect_Density = [
        (pred[j] / testingcode[j]) + 100000 if pred[j] > 0.5 else (pred[j] / testingcode[j]) - 100000 if
        testingcode[j] != 0 else 0 for j in range(len(pred))]

    defect_prob = []
    for i in range(len(defect_Density)):
        defect_prob.append(defect_Density[i] * testingcode[i])

    _, _, _, _, pofb = PerformanceMeasure(testing_data_value, defect_prob, testingcode, 0.2, 'density',
                                          'loc').Performance()

    return pofb


def backward_search(dataset_train, dataset_test, classifier):
    """
    apply backward search for wrapper-based feature subset selection
    :param dataset_train:
    :param dataset_test:
    :param classifier:
    :return:
    """
    training_data_x, training_data_y = spilt_data_classification(dataset_train)
    testing_data_x, testing_data_y = spilt_data_classification(dataset_test)
    # extract real number of defect and LOC
    testing_data_value = dataset_test.loc[:, 'bug'].tolist()
    testingcode = testing_data_x.iloc[:, 10].tolist()

    # get classifers
    model = classifier_name[classifier](training_data_x, training_data_y)
    preds = model.predict_proba(testing_data_x)
    pofb = Pofb(preds, testingcode, testing_data_value)
    best_pofb = pofb
    column = -1
    old_column = column
    dropped_cloumn = []
    selected_column = []
    new_training_data_x = training_data_x
    new_testing_data_x = testing_data_x
    # selected_column.append(column)

    for i in range(training_data_x.shape[1] - 1):  # 19 epoch
        print(new_training_data_x.columns.values.tolist())
        for j in new_training_data_x.columns.values.tolist():  # number of features of the current epoch
            tmp_training_data_x = new_training_data_x.drop(j, axis=1)
            tmp_testing_data_x = new_testing_data_x.drop(j, axis=1)
            # print(tmp_training_data_x.columns.values)

            new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
            new_preds = new_model.predict_proba(tmp_testing_data_x)
            new_pofb = Pofb(new_preds, testingcode, testing_data_value)
            if new_pofb >= best_pofb:
                best_pofb = new_pofb
                column = j

        if old_column != column:
            new_training_data_x = new_training_data_x.drop(column, axis=1)
            new_testing_data_x = new_testing_data_x.drop(column, axis=1)
            dropped_cloumn.append(column)
            old_column = column
            if i == training_data_x.shape[1] - 2:
                for p in training_data_x.columns.values.tolist():
                    if p not in dropped_cloumn:
                        selected_column.append(p)
                # print(selected_column)
                return dropped_cloumn, selected_column, best_pofb

        else:
            for k in training_data_x.columns.values.tolist():
                if k not in dropped_cloumn:
                    selected_column.append(k)
            # print(selected_column)
            return dropped_cloumn, selected_column, best_pofb


def forward_search(dataset_train, dataset_test, classifier):
    """
    apply forkward search for wrapper-based feature subset selection
    :param dataset_train:
    :param dataset_test:
    :param classifier:
    :return:
    """
    training_data_x, training_data_y = spilt_data_classification(dataset_train)
    testing_data_x, testing_data_y = spilt_data_classification(dataset_test)
    # extract real number of defect and LOC
    testing_data_value = dataset_test.loc[:, 'bug'].tolist()
    testingcode = testing_data_x.iloc[:, 10].tolist()

    best_pofb = 0
    column = -1

    # new_training_data_x = training_data_x
    # new_testing_data_x = testing_data_x

    for i in range(training_data_x.shape[1]):
        tmp_training_data_x = training_data_x[i].to_frame()
        tmp_testing_data_x = testing_data_x[i].to_frame()
        # print(tmp_training_data_x.columns.values)
        new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
        new_preds = new_model.predict_proba(tmp_testing_data_x)
        new_pofb = Pofb(new_preds, testingcode, testing_data_value)
        if new_pofb >= best_pofb:
            best_pofb = new_pofb
            column = i
            new_training_data_x = tmp_training_data_x
            new_testing_data_x = tmp_testing_data_x

    old_column = column
    selected_cloumn = []
    dropped_column = []
    selected_cloumn.append(column)

    for i in range(training_data_x.shape[1] - 1):
        candidate_columns = []
        for col in training_data_x.columns.values.tolist():
            if col not in new_training_data_x.columns.values.tolist():
                candidate_columns.append(col)
        # print(candidate_columns)
        print(new_training_data_x.columns.values.tolist())
        for j in candidate_columns:
            tmp_training_data_x = pd.concat([new_training_data_x, training_data_x[j].to_frame()], axis=1)
            tmp_testing_data_x = pd.concat([new_testing_data_x, testing_data_x[j].to_frame()], axis=1)
            # print(tmp_training_data_x.columns.values)
            new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
            new_preds = new_model.predict_proba(tmp_testing_data_x)
            new_pofb = Pofb(new_preds, testingcode, testing_data_value)
            if new_pofb >= best_pofb:
                best_pofb = new_pofb
                column = j

        if old_column != column:
            new_training_data_x = pd.concat([new_training_data_x, training_data_x[column].to_frame()], axis=1)
            new_testing_data_x = pd.concat([new_testing_data_x, testing_data_x[column].to_frame()], axis=1)
            selected_cloumn.append(column)
            old_column = column
            if i == training_data_x.shape[1] - 2:
                for p in training_data_x.columns.values.tolist():
                    if p not in selected_cloumn:
                        dropped_column.append(p)
                print(selected_cloumn)
                return dropped_column, selected_cloumn, best_pofb
        else:
            for k in training_data_x.columns.values.tolist():
                if k not in selected_cloumn:
                    dropped_column.append(k)
            print(selected_cloumn)
            return dropped_column, selected_cloumn, best_pofb


if __name__ == '__main__':
    classifier_name = {
        'DF': deep_forest,
        'XGB': xgboost,
        'RF': random_forest_classifier,
        'ADB': adaboost,
    }
    classifiers = ['DF', 'XGB', 'RF', 'ADB']

    performence = []
    folder_path = '../CrossversionData'

    # count = 1
    # open directory
    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)
                for root, dirs, files, in os.walk(dir_path):
                    if files[0][-7:-4] < files[1][-7:-4]:
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])
                    else:
                        file_path_train = os.path.join(dir_path, files[1])
                        file_path_test = os.path.join(dir_path, files[0])

                    # print('Now reading the dataset:{}'.format(count))
                    # print('train version', files[0][-7:-4])
                    # print('test version', files[1][-7:-4])
                    # print(files[0][-7:-4], '<', files[1][-7:-4])
                    # print('train path', file_path_train)
                    # print('test path', file_path_test)
                    # print('***********************************')
                    # count += 1

                    # read training and testing datasets
                    dataset_train = pd.read_csv(file_path_train)
                    dataset_test = pd.read_csv(file_path_test)
                    with open("../Feature/" + dir + '/WrapperSubSetSelection.txt', 'w') as f:
                        result = []
                        for classifier in classifiers:
                            # print(classifier)
                            _, selected_cloumn1, _ = backward_search(dataset_train, dataset_test, classifier)
                            f.write('wrapperSubset' + classifier + 'Pofb' + '-Backward ')
                            for col in selected_cloumn1:
                                f.write(str(col) + ' ')
                            f.write('20\n')

                            _, selected_cloumn2, _ = forward_search(dataset_train, dataset_test, classifier)
                            # print(selected_cloumn2)
                            f.write('wrapperSubset' + classifier + 'Pofb' + '-Forward ')
                            for col in selected_cloumn2:
                                f.write(str(col) + ' ')
                            f.write('20\n')
                    f.close()
