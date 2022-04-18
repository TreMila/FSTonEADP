package cn.lwp.whut.feature;

import weka.attributeSelection.*;

/**
 * @Description: Filter-based feature ranking methods
 * @Source: JDK 1.8
 * @Author: LuWanpeng
 * @Date: 2022-04-18 19:45
 * @Since: version 1.0.0
 **/
public class FilterFeatureRanking {

    /**
     * Statistics-based methods
     */
    public static ASEvaluation chiSquare() {
        return new ChiSquaredAttributeEval();
    }

    public static ASEvaluation correlation() {
        return new CorrelationAttributeEval();
    }

    public static ASEvaluation clusteringVariation() {
        return new CVAttributeEval();
    }


    /**
     * Probability-based methods
     */
    public static ASEvaluation probabilisticSignificance() {
        return new SignificanceAttributeEval();
    }

    public static ASEvaluation infoGain() {
        return new InfoGainAttributeEval();
    }

    public static ASEvaluation gainRatio() {
        return new GainRatioAttributeEval();
    }

    public static ASEvaluation symmetrical() {
        return new SymmetricalUncertAttributeEval();
    }

    /**
     * Instance-based methods
     */
    public static ASEvaluation reliefF() {
        return new ReliefFAttributeEval();
    }

    public static ASEvaluation reliefFWeight() {
        ReliefFAttributeEval reliefWeighted = new ReliefFAttributeEval();
        reliefWeighted.setWeightByDistance(true);
        return reliefWeighted;
    }

    /**
     * Classifier-based methods
     */
    public static ASEvaluation oneR() {
        return new OneRAttributeEval();
    }

    public static ASEvaluation svm() {
        return new SVMAttributeEval();
    }

}
