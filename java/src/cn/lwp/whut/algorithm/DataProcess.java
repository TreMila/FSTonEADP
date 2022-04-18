package cn.lwp.whut.algorithm;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;


/**
 * @Description: Processing the data
 * @Source: JDK 1.8
 * @Author: LuWanpeng
 * @Date: 2022-04-18 19:45
 * @Since: version 1.0.0
 **/
public class DataProcess {
    /**
     * Normalize and nominal the data
     *
     * @param instances
     * @return
     * @throws Exception
     */
    public static Instances normalizeAndNominalData(Instances instances) throws Exception {

        Filter filter = new Normalize();
        filter.setInputFormat(instances);
        Instances normalizedIns = Filter.useFilter(instances, filter);

        NumericToNominal nominal = new NumericToNominal();
        nominal.setInputFormat(normalizedIns);
        nominal.setAttributeIndices("last");
        Instances nominalIns = Filter.useFilter(normalizedIns, nominal);

        return nominalIns;
    }
}
