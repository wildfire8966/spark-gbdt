package com.weibo.yuanye.lr;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

/**
 * Created by yuanye8 on 2016/11/18.
 */
public class LrTrain {
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: <corps_path> <txt_model>");
        }
        String inputPath = args[0];
        String txtModel = args[1];

        SparkConf conf = new SparkConf();
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc.sc(), inputPath).toJavaRDD();

        JavaRDD[] splits = data.randomSplit(new double[]{0.9, 0.1});
        JavaRDD<LabeledPoint> traindata = splits[0].cache();
        JavaRDD<LabeledPoint> testdata = splits[1];

        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(traindata.rdd());

        JavaRDD<Tuple2<Double, Double>> predictionAndLabels = testdata.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
            public Tuple2<Double, Double> call(LabeledPoint p) throws Exception {
                Double prediction = model.predict(p.features());
                return new Tuple2<Double, Double>(prediction, p.label());
            }
        });

/*      BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(predictionAndLabels.rdd());
        double auc = metrics.areaUnderROC();*/

        Double testAcc = Double.valueOf(1.0 * predictionAndLabels.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            public Boolean call(Tuple2<Double, Double> pl) throws Exception {
                return pl._1().equals(pl._2());
            }
        }).count() / testdata.count());

        System.out.println("Test Accuracy = " + testAcc);

        //保存模型
        double[] weights = model.weights().toArray();
        FileSystem hdfs = FileSystem.get(new Configuration());
        Path txtModelPath = new Path(txtModel);
        if (hdfs.exists(txtModelPath) && hdfs.isFile(txtModelPath)) {
            hdfs.delete(txtModelPath, true);
        }
        BufferedWriter fsout = new BufferedWriter(
                new OutputStreamWriter(
                        hdfs.create(txtModelPath)));
        fsout.write("Test Accuracy = " + String.valueOf(testAcc) + "\n");
        fsout.write("Learned classification LR model:\n");
        for (double weight : weights) {
            fsout.write(String.valueOf(weight) + "\n");
        }
        fsout.flush();
        fsout.close();
    }
}
