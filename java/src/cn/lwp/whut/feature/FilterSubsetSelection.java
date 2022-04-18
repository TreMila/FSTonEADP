package cn.lwp.whut.feature;

import weka.attributeSelection.*;

/**
 * @Description: Filter-based subset Selection methods
 * @Source: JDK 1.8
 * @Author: LuWanpeng
 * @Date: 2022-04-18 19:45
 * @Since: version 1.0.0
 **/
public class FilterSubsetSelection {

    /**
     * Correlation-based feature subset selection
     */
    public static ASEvaluation CfsSubset() {
        return new CfsSubsetEval();
    }


    /**
     * Consistency-based feature subset selection
     */
    public static ASEvaluation consistencySubset() {
        return new ConsistencySubsetEval();
    }
}
