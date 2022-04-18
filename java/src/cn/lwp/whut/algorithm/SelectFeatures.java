package cn.lwp.whut.algorithm;

import cn.lwp.whut.classifiers.ClassificationModels;
import cn.lwp.whut.feature.FilterFeatureRanking;
import cn.lwp.whut.feature.FilterSubsetSelection;
import cn.lwp.whut.feature.SearchMethod;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * @Description: Select features with Feature selection methods
 * @Source: JDK 1.8
 * @Author: LuWanpeng
 * @Date: 2022-04-18 19:52
 * @Since: version 1.0.0
 **/
public class SelectFeatures {
	
	/**
	 * write feature selection results into txt
	 * @param FilePath
	 * @param content
	 */
	public static void writeFile2Txt(String FilePath, String content) {
		File file;
		FileOutputStream fop = null;
		
		try {
			file = new File(FilePath);
			fop = new FileOutputStream(file, true);
			if (!file.exists()) {
				file.createNewFile();
			}
			byte[] contentInBytes = content.getBytes();
			fop.write(contentInBytes);
			fop.flush();
			fop.close();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (fop != null) {
					fop.close();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	

	/**
	 * FeatureRanking + Classifier
	 * @param train
	 * @param test
	 * @param featurePath
	 * @param predsPath
	 * @throws Exception
	 */
	public static void applyFeatureRankingClassifier(Instances train,
			Instances test, String featurePath, String predsPath) throws Exception {
		
		Method[] featureSelectionMethods = FilterFeatureRanking.class.getDeclaredMethods();
		Method[] classificationMethods = ClassificationModels.class.getDeclaredMethods();
		
		File file1 = new File(featurePath);
		if(!file1.exists()){
			file1.mkdirs();
		}
		
		File file2 = new File(predsPath);
		if(!file2.exists()){
			file2.mkdirs();
		}
		
		for (Method featureSelectionMethod : featureSelectionMethods) {
			System.out.println("Current feature selection methods:" + featureSelectionMethod.getName() + "...");
			ASEvaluation evaluation = (ASEvaluation) featureSelectionMethod.invoke(null);
			Ranker ranker = rankSearch();
			Map<List<Instances>, List<Integer>> map = selectFeature(evaluation, ranker, train, test);
			List<Instances> instances = null;
			List<Integer> attributesLists = null;
			
			for (Map.Entry<List<Instances>, List<Integer>> vo : map.entrySet()) {
				instances = vo.getKey();
				attributesLists = vo.getValue();
			}

			String feature = "";
			for (Integer index : attributesLists) {
				feature = feature + " " + Integer.toString(index);
			}
			String line = featureSelectionMethod.getName() + " " + feature + "\n";
			/**
			 * line: featureSelectionMethod column1 column2 column3...
			 */
			writeFile2Txt(featurePath + "\\FeatureRanking.txt", line);
			System.out.println("Feature selection method:" + featureSelectionMethod.getName() + " selected feature columns:" + feature);

			Instances trainIns = instances.get(0);
			Instances testIns = instances.get(1);

			for (Method classificationMethod : classificationMethods) {
				Classifier classifier = (Classifier) classificationMethod.invoke(null);
				System.out.println("Feature selection method:" + featureSelectionMethod.getName() + " with classifier:"
						+ classificationMethod.getName() + " training...");
				classifier.buildClassifier(trainIns);
				int len = testIns.toArray().length;
			
				for(int i = 0; i < len ;i++){
					double pred = classifier.distributionForInstance(testIns.instance(i))[1];
					String content = String.valueOf(pred) + "\n";
					File file3 = new File(predsPath + "\\FeatureRanking\\" + featureSelectionMethod.getName());
					if(!file3.exists()) {
						file3.mkdirs();
					}
					writeFile2Txt(predsPath + "\\FeatureRanking\\" + featureSelectionMethod.getName() + 
							"\\" + classificationMethod.getName() + ".txt", content);		
				}
			}
		}
		System.out.println("FeatureRanking finish!");
	}


	/**
	 * FeatureSubsetSelection + Classifier
	 * @param train
	 * @param test
	 * @param featurePath
	 * @param predsPath
	 * @throws Exception
	 */
	public static void applySubSetSelectionClassifier(Instances train,
			Instances test, String featurePath, String predsPath) throws Exception {

		Method[] subSetSelectionMethods = FilterSubsetSelection.class.getDeclaredMethods();
		Method[] classificationMethods = ClassificationModels.class.getDeclaredMethods();
		Method[] searchMethods = SearchMethod.class.getDeclaredMethods();
		
		File file1 = new File(featurePath);
		if(!file1.exists()){
			file1.mkdirs();
		}
		
		File file2 = new File(predsPath);
		if(!file2.exists()){
			file2.mkdirs();
		}
		
		
		for (Method subSetSelectionMethod : subSetSelectionMethods) {
			
			System.out.println("Current feature selection method£º" + subSetSelectionMethod.getName() + "...");
		
			ASEvaluation evaluation = (ASEvaluation) subSetSelectionMethod.invoke(null);
			for (Method searchMethod : searchMethods) {
				ASSearch search = (ASSearch) searchMethod.invoke(null);
				System.out.println("Feature selection method and search method:" + subSetSelectionMethod.getName() + "-" + searchMethod.getName() + "...");
				
				Map<List<Instances>, List<Integer>> map = selectFeature(evaluation, search, train, test);
				List<Instances> instances = null;
				List<Integer> attributesLists = null;
				
				for (Map.Entry<List<Instances>, List<Integer>> vo : map.entrySet()) {
					instances = vo.getKey();
					attributesLists = vo.getValue();
				}

				String feature = "";
				for (Integer index : attributesLists) {
					feature = feature + " " + Integer.toString(index);
				}
				String line = subSetSelectionMethod.getName() + "-" + searchMethod.getName() + " " + feature + "\n";
				
				/**
				 * line: SubSetSelectionionMethod column1 column2 column3...
				 */
				writeFile2Txt(featurePath + "\\SubSetSelection.txt", line);
				System.out.println("feature selection and search method:" + subSetSelectionMethod.getName() + "-" + searchMethod.getName() + 
						" selected feature columns:" + feature);

				Instances trainIns = instances.get(0);
				Instances testIns = instances.get(1);

				for (Method classificationMethod : classificationMethods) {
					Classifier classifier = (Classifier) classificationMethod.invoke(null);
					System.out.println("feature selection and search method:" + subSetSelectionMethod.getName() + "-" + searchMethod.getName() + 
							" with classifier" + classificationMethod.getName() + " training...");

					classifier.buildClassifier(trainIns);
					int len = testIns.toArray().length;
			
					for(int i = 0; i < len ;i++){
						double pred = classifier.distributionForInstance(testIns.instance(i))[1];
						String content = String.valueOf(pred) + "\n";
						File file3 = new File(predsPath + "\\SubSetSelection\\" + subSetSelectionMethod.getName() + 
								"-" + searchMethod.getName());
						if(!file3.exists()) {
							file3.mkdirs();
						}
						writeFile2Txt(predsPath + "\\SubSetSelection\\" + subSetSelectionMethod.getName() + 
								"-" + searchMethod.getName() + "\\" + classificationMethod.getName() + ".txt", content);		
					}
				}
			}
		}
		System.out.println("SubsetSelection finish!");

	}


	/**
	 * Select the feature
	 * @param m_Evaluator
	 * @param search
	 * @param instances
	 * @param test
	 * @return
	 * @throws Exception
	 */
	public static Map<List<Instances>, List<Integer>> selectFeature(ASEvaluation m_Evaluator, ASSearch search,
			Instances instances, Instances test) throws Exception {
		Map<List<Instances>, List<Integer>> map = new HashMap<List<Instances>, List<Integer>>();
		AttributeSelection attributeSelection = new AttributeSelection();
		attributeSelection.setEvaluator(m_Evaluator);
		attributeSelection.setSearch(search);
		attributeSelection.SelectAttributes(instances);

		int[] attributesIdx = attributeSelection.selectedAttributes();

		List<Integer> attributesLists = new ArrayList<>();
		for (int i = 0; i < attributesIdx.length; i++) {
			attributesLists.add(attributesIdx[i]);
		}
//        System.out.println(attributesLists);
		Instances newIns = new Instances(instances);
		for (int i = newIns.numAttributes() - 1; i > -1; i--) { // delete from back to forward
			if (!attributesLists.contains(i))
				newIns.deleteAttributeAt(i);
		}

		// System.out.println(newIns);

		Instances newTestIns = new Instances(test);
		for (int i = newTestIns.numAttributes() - 1; i > -1; i--) { // delete from back to forward
			if (!attributesLists.contains(i))
				newTestIns.deleteAttributeAt(i);
		}

		List<Instances> trainTestIns = new ArrayList<>();
		trainTestIns.add(newIns);
		trainTestIns.add(newTestIns);

		map.put(trainTestIns, attributesLists);

		return map;
	}

	
	/**
	 * None Feature + Classifier
	 * @param train
	 * @param test
	 * @param featurePath
	 * @param predsPath
	 * @throws Exception
	 */
	public static void applyNoneFeatureClassifier(Instances train,
			Instances test, String featurePath, String predsPath)throws Exception {

		Method[] classificationMethods = ClassificationModels.class.getDeclaredMethods();
		File file1 = new File(featurePath);
		if(!file1.exists()){
			file1.mkdirs();
		}
		
		File file2 = new File(predsPath);
		if(!file2.exists()){
			file2.mkdirs();
		}
		
		for (Method classificationMethod : classificationMethods) {
			Classifier classifier = (Classifier) classificationMethod.invoke(null);
			
			System.out.println("None with classifer:" + classificationMethod.getName() + " training...");
			
			classifier.buildClassifier(train);
			int len = test.toArray().length;
			
			for(int i = 0; i < len ;i++){
				double pred = classifier.distributionForInstance(test.instance(i))[1];
				String content = String.valueOf(pred) + "\n";
				File file3 = new File(predsPath + "\\NoneFeature");
				if(!file3.exists()) {
					file3.mkdirs();
				}
				writeFile2Txt(predsPath + "\\NoneFeature\\" + classificationMethod.getName() + ".txt", content);		
			}		
		}
		System.out.println("None finish!");
	}

	
	/**
	 * Search for Feature Ranking
	 *
	 * @return
	 */
	public static Ranker rankSearch() {
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(4);
		return ranker;
	}
	
	
	/**
	 * use feature selection methods
	 * @param trainpath
	 * @param testpath
	 * @param featurePath
	 * @param predsPath
	 * @param featureMethod
	 * @throws Exception
	 */
	public static void FeatureSelection(String trainpath, String testpath, 
			 String featurePath, String predsPath, String featureMethod)  throws Exception { 

	        Instances ins1 = null;
			Instances ins2 = null;

			File file1 = new File(trainpath);
			File file2 = new File(testpath);
					
			ArffLoader loader1 = new ArffLoader();
			loader1.setFile(file1);
			ins1 = loader1.getDataSet();
			ins1.setClassIndex(ins1.numAttributes() - 1);

			ArffLoader loader2 = new ArffLoader();
			loader2.setFile(file2);
			ins2 = loader2.getDataSet();
			ins2.setClassIndex(ins2.numAttributes() - 1);

			Instances train = new Instances(ins1);
			Instances test = new Instances(ins2);

			train = DataProcess.normalizeAndNominalData(train);
			test = DataProcess.normalizeAndNominalData(test);

	        switch (featureMethod){
	            case ("Ranking"):
	            	applyFeatureRankingClassifier(train, test, featurePath, predsPath);
	                break;
	            case ("Subset"):
	            	applySubSetSelectionClassifier(train, test, featurePath, predsPath);
	            	break;
	            case ("None"):
	            	applyNoneFeatureClassifier(train, test, featurePath, predsPath);
	            	break;
	            default:
	                System.out.println("Not choose");
	        }

	}

}
