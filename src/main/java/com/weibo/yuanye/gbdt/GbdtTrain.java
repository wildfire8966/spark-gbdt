package com.weibo.yuanye.gbdt;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by yuanye8 on 2016/11/16.
 */
public class GbdtTrain {
    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            System.out.println("Usage: <train-corps> <model-path> <txt-model-path>");
            System.exit(-1);
        }

        String trainCorps = args[0];
        String modelPath = args[1];
        String txtModel = args[2];

        SparkConf sc = new SparkConf();
        JavaSparkContext jsc = new JavaSparkContext(sc);

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), trainCorps).toJavaRDD();
        JavaRDD[] splits = data.randomSplit(new double[] {0.9, 0.1});
        JavaRDD<LabeledPoint> trainData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        trainData.map(new Function<LabeledPoint, String>() {
            int index = 1;
            public String call(LabeledPoint p) throws Exception {
                StringBuilder sb = new StringBuilder();
                int label = (int)p.label();
                sb.append(label).append(" ");
                for (double d : p.features().toArray()) {
                    sb.append(index).append(":").append(d).append(" ");
                    index ++;
                }
                index = 1;
                return sb.deleteCharAt(sb.length() - 1).toString();
            }
        }).saveAsTextFile("/user/weibo_bigdata_dm/yuanye/data/gbdt/traindata");

        testData.map(new Function<LabeledPoint, String>() {
            int index = 1;
            public String call(LabeledPoint p) throws Exception {
                StringBuilder sb = new StringBuilder();
                int label = (int)p.label();
                sb.append(label).append(" ");
                for (double d : p.features().toArray()) {
                    sb.append(index).append(":").append(d).append(" ");
                    index ++;
                }
                index = 1;
                return sb.deleteCharAt(sb.length() - 1).toString();
            }
        }).saveAsTextFile("/user/weibo_bigdata_dm/yuanye/data/gbdt/testdata");

        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(30);
        boostingStrategy.getTreeStrategy().setNumClasses(2);
        boostingStrategy.getTreeStrategy().setMaxDepth(3);

        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        final GradientBoostedTreesModel model = GradientBoostedTrees.train(trainData, boostingStrategy);

        //predict推测出的结果为0，1
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>(){
            public Tuple2<Double, Double> call(LabeledPoint p) throws Exception {
                return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
            }
        });

        Double testAcc = Double.valueOf(1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            public Boolean call(Tuple2<Double, Double> pl) throws Exception {
                return pl._1().equals(pl._2());
            }
        }).count() / testData.count());

        //平台1.3.1，1.5.1，1.6.1版本均出错，OutOfMemory
        //model.save(jsc.sc(), modelPath);
        System.out.println("Test Accuracy: " + testAcc);
        System.out.println("Learned classification GBT model:\n" + model.toDebugString());

        //保存txt模型
        FileSystem hdfs = FileSystem.get(new Configuration());
        Path txtModelPath = new Path(txtModel);
        if (hdfs.exists(txtModelPath) && hdfs.isFile(txtModelPath)) {
            hdfs.delete(txtModelPath, true);
        }
        BufferedWriter fsout = new BufferedWriter(
                                    new OutputStreamWriter(
                                            hdfs.create(txtModelPath)));
        fsout.write("Test Accuracy: " + String.valueOf(testAcc) + "\n");
        fsout.write("Learned classification GBT model:\n" + model.toDebugString());
        fsout.flush();
        fsout.close();
    }
}
