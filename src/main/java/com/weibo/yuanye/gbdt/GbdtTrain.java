package com.weibo.yuanye.gbdt;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.glassfish.grizzly.servlet.ver25.String;

/**
 * Created by yuanye8 on 2016/11/16.
 */
public class GbdtTrain {
    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Usage: <train-corps> <model-path> <txt-model-path>");
            System.exit(-1);
        }

        String trainCorps = args[0];
        String modelPath = args[1];
        String txtModelPath = args[2];

        SparkConf sc = new SparkConf();
        JavaSparkContext jsc = new JavaSparkContext(sc);

    }
}
