# coding=utf-8

import os


class configuration_file():
    def __init__(self):
        # root path
        self.rootpath = os.path.dirname(os.getcwd())
        # directory to save original data
        self.dataPath = os.path.join(self.rootpath, "CrossversionData")
        # directory to save results of performance measure
        self.savePerformanceMeasureResult_dir = os.path.join(self.rootpath, "PerformanceMeasureResult")
        # directory to save features selected
        self.saveFeature_dir = os.path.join(self.rootpath, 'Feature')

        pass

    def getrootpath(self):
        return self.rootpath
