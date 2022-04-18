package cn.lwp.whut.main;

import cn.lwp.whut.algorithm.SelectFeatures;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * @Description: Main
 * @Source: JDK 1.8
 * @Author: LuWanpeng
 * @Date: 2022-04-18 19:45
 * @Since: version 1.0.0
 **/
public class Main {
	public static void main(String[] args) throws Exception {
		SimpleDateFormat formatter = new SimpleDateFormat("dd-MM-yyyy HH:mm:ss");
		Date startTime = new Date(System.currentTimeMillis());
		System.out.println(formatter.format(startTime));
		
		
		String csvPath = "D:\\python\\Feature_Select\\CrossversionData";
		String arffPath = "D:\\python\\Feature_Select\\ArffData";
		File file = new File(csvPath);
		String[] dirList = file.list();
		for (int i = 0; i < dirList.length; i++) {
			File dir = new File(csvPath + "\\" + dirList[i]);
			
			String featurePath = "D:\\python\\Feature_Select\\Feature\\" + dirList[i];
			String predsPath = "D:\\python\\Feature_Select\\predDistribution\\" + dirList[i];
//			
//			System.out.println(featurePath);
//			System.out.println(predsPath);
			
			String[] fileList = dir.list();

			File csvTrain = new File(csvPath + "\\" + dirList[i] + "\\" + fileList[0]);
			File csvTest = new File(csvPath + "\\" + dirList[i] + "\\" + fileList[1]);

			String trainName = csvTrain.getName().split(".csv")[0];
			String testName = csvTest.getName().split(".csv")[0];
			
			String trainPath = arffPath + "\\" + trainName + ".arff";
			String testPath = arffPath + "\\" + testName + ".arff";
			System.out.println(trainPath);
			System.out.println(testPath);
			
			/**
			 * here 'Ranking' can be replaced with 'Subset' and 'None'
			 */
			SelectFeatures.FeatureSelection(trainPath, testPath, featurePath, predsPath, "Ranking");
		}

		Date endTime = new Date(System.currentTimeMillis());
		System.out.println(formatter.format(endTime));
	}
}

