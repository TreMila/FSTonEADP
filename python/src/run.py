# coding=utf-8
import os

from src.Processing import *
from src.configuration_file import configuration_file
from src.classification import *
from src.PerformanceMeasure import PerformanceMeasure

import warnings
warnings.filterwarnings('ignore')


def new_data(data, feature):
    """
    Generate new training and testing datasets with selected features
    :param data: original datasets
    :param feature: selected features
    :return: new training and testing datasets
    """
    data = np.array(data)
    index = []
    # delete the last column : bug
    for i in feature:
        if int(i) != 20 and i != '':
            index.append(int(i))
    n_data = data[:, index]
    return n_data


def SelectFeature(feature_path, featureMethod):
    """
    get selected features
    :param feature_path:
    :param featureMethod:
    :return:
    """
    if featureMethod != 'NoneFeature':
        res = {}
        with open(feature_path + '/' + featureMethod + '.txt', 'r') as f1:
            for line in f1:
                line = line.strip().split()
                res[line[0]] = line[1:]
    else:
        res = {'nofeature': [20]}
        for i in range(20):
            res['nofeature'].append(i)
    return res


def CalculateDensity(classifier, X, Y, testX, testingcodeN):
    """
    calculate defect density
    :param classifier:
    :param X: features in training datasets
    :param Y: label in training datasets
    :param testX: features in testing datasets
    :param testingcodeN: LOC in testing datasets
    :return:
    """
    model = classifier_name[classifier](X, Y)
    preds = model.predict_proba(testX)
    pred = [p[1] for p in preds]
    defect_Density = [
        (pred[j] / testingcodeN[j]) + 100000 if pred[j] > 0.5 else (pred[j] / testingcodeN[j]) - 100000 if
        testingcodeN[j] != 0 else 0 for j in range(len(pred))]
    defect_prob = []
    for i in range(len(defect_Density)):
        defect_prob.append(defect_Density[i] * testingcodeN[i])
    return defect_prob


if __name__ == '__main__':
    folder_path = configuration_file().dataPath + '/'
    header = ["dataset", "FeatureMethod", "Classifier", "precision", "recall", "pofb", "pmi", "ifma", "popt"]
    resultlist = []  # rows: results of performance measures
    featuremethods = ['FeatureRanking', 'SubsetSelection', 'WrapperSubSetSelection', 'NoneFeature']
    classifiers = ['ADB', 'DF', 'DT', 'KNN', 'LR', 'NB', 'RF', 'XGB']
    classifier_name = {
        'ADB': adaboost,
        'DF': deep_forest,
        'DT': decision_tree_classifier,
        'KNN': knn,
        'LR': logistic_regression_classifier,
        'NB': naive_bayes_classifier,
        'RF': random_forest_classifier,
        'XGB': xgboost
    }

    # append header into the resultlist
    resultlist.append(header)
    # count = 1

    # open directory
    for root, dirs, files, in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                dir_path = os.path.join(thisroot, dir)
                for root, dirs, files, in os.walk(dir_path):
                    # judge the order of the version of files
                    if (files[0][-7:-4] < files[1][-7:-4]):
                        file_path_train = os.path.join(dir_path, files[0])
                        file_path_test = os.path.join(dir_path, files[1])
                        trainingfile = files[0]
                        testingfile = files[1]
                    else:
                        file_path_train = os.path.join(dir_path, files[1])
                        file_path_test = os.path.join(dir_path, files[0])
                        trainingfile = files[1]
                        testingfile = files[0]

                    # print('Now reading the dataset:{}'.format(count))
                    # count += 1
                    # print('train version', files[0][-7:-4])
                    # print('test version', files[1][-7:-4])
                    # print(files[0][-7:-4], '<', files[1][-7:-4])
                    # print('train path', file_path_train)
                    # print('test path', file_path_test)
                    # print('***********************************')

                    # read training and testing datasets
                    dataset_train = pd.read_csv(file_path_train)
                    dataset_test = pd.read_csv(file_path_test)

                    # split features and labels in training and testing datasets
                    training_data_x, training_data_y = spilt_data(dataset_train)
                    testing_data_x, testing_data_y = spilt_data(dataset_test)
                    # extract LOC
                    testingcodeN = testing_data_x.iloc[:, 10].tolist()
                    # extract real number of defect
                    testing_data_value= dataset_test.loc[:, 'bug'].tolist()

                    training_data = training_data_x
                    testing_data = testing_data_x
                    for featuremethod in featuremethods:
                        print("Family of feature selection methods:" + featuremethod)
                        featureresult = SelectFeature(configuration_file().saveFeature_dir + '/' + dir, featuremethod)
                        methods = featureresult.keys()
                        for method in methods:
                            selectedFeature = featureresult[method]
                            # simplify training and testing datasets
                            if len(selectedFeature) != 1:
                                training_data_x = new_data(training_data, selectedFeature)
                                testing_data_x = new_data(testing_data, selectedFeature)

                                for classifier in classifiers:
                                    print(method + '-' + classifier)
                                    defect_prob = CalculateDensity(classifier, training_data_x, training_data_y, testing_data_x, testingcodeN)

                                    # calculate performance measures
                                    Precisionx, Recallx, IFMA, PMI, Pofb= PerformanceMeasure(testing_data_value, defect_prob, testingcodeN, 0.2, 'density', 'loc').Performance()

                                    Popt = PerformanceMeasure(testing_data_value, defect_prob, testingcodeN, 0.2, 'density', 'loc').POPT()

                                    Results = []
                                    dataset = trainingfile + testingfile

                                    # append performance measures
                                    Results.append(dataset)
                                    Results.append(method)
                                    Results.append(classifier)

                                    Results.append(Precisionx)
                                    Results.append(Recallx)
                                    Results.append(Pofb)
                                    Results.append(PMI)
                                    Results.append(IFMA)
                                    Results.append(Popt)

                                    resultlist.append(Results)

                            # when the number of selected feaures = 0
                            else:
                                for classifier in classifiers:
                                    Results = []
                                    dataset = trainingfile + testingfile

                                    Results.append(dataset)
                                    Results.append(method)
                                    Results.append(classifier)

                                    Results.append(0)
                                    Results.append(0)
                                    Results.append(0)
                                    Results.append(0)
                                    Results.append(0)
                                    Results.append(0)

                                    resultlist.append(Results)
                                print(dir + ":" + method + ":" + "Too few features!")

    # save results
    result_path = configuration_file().savePerformanceMeasureResult_dir
    result_csv_name = "result.xlsx"
    result_path = os.path.join(result_path, result_csv_name)
    write_excel(result_path, resultlist)