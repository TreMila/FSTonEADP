package cn.lwp.whut.feature;


import weka.attributeSelection.BestFirst;
import weka.attributeSelection.GreedyStepwise;

/**
 * @Description: Search methods
 * @Source: JDK 1.8
 * @Author: LuWanpeng
 * @Date: 2022-04-18 19:45
 * @Since: version 1.0.0
 **/
public class SearchMethod {

    private SearchMethod() {}

    public static BestFirst bestFirst() {
        return new BestFirst();
    }

    public static GreedyStepwise greedyStepwise() {
        GreedyStepwise greedyStepwise =  new GreedyStepwise();
        greedyStepwise.setNumToSelect(4);
        greedyStepwise.setSearchBackwards(true);
        return greedyStepwise;
    }

}
