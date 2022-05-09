# FS
##  java: java codes  
* ArffData: Training and testing datasets
* lib: The jar packages that the algorithm needs to call (weka.jar, SVMAttributeEval.jar, chiSquaredAttributeEval.jar, etc)
* src: Java code files
  * cn.lwp.whut.algorithm
    * DataProcess.java: Process the datasets
    * SelectionFeatures.java: Select features with filter-based feature ranking methods, filter-based feature subset selection methods
  * cn.lwp.whut.feature
    * FilterFeatureRanking.java: Filter-based feature ranking methods
    * FilterSubsetSelection.java: Filter-based feature subset selection methods
    * SearchMethod.java: Search strategies for Filter-based feature subset selection methods
  * cn.lwp.whut.main
    * Mian.java: The main method to run the whole java codes
## python: python codes 
* CrossVersionData: Training and testing datasets pairs
* lib: The packages that the algorithm needs to call (sklearn, numpy, pandas, etc)
  * packages.zip: The compressed packages (sklearn, numpy, pandas, etc). **It is recommended to download this 'packages.zip' separately here.**
* src: Python code files
  * configuration.py: Store the file paths
  * classifiers.py: The classification models we studied
  * PerformanceMeasure.py: Calculate the performance measures
  * Processing.py: Process the datasets
  * wrapper_subset_selection.py: Define the wrapper-based feature subset selection methods
  * run.py: the main method to run the whole python codes
