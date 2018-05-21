package org.wekaservice.logic;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.JRip;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaMiner {
    private final Logger LOGGER = LoggerFactory.getLogger(getClass().getName());

    public String buildModel(String dataName, String algorithm) throws Exception {
	String modelFileName = "";
	LOGGER.info(dataName + " ------ " + algorithm);

	String trainFile = dataName + ".arff";
	String testFile = dataName + "_test.arff";
	String predictFile = dataName + "_predict.arff";

	DataSource source = new DataSource(trainFile);
	Instances dataset = source.getDataSet();
	dataset.setClassIndex(dataset.numAttributes() - 1);

	// create classifier/algorithm
	// if (algorithm.equals("JRip")) {
	switch (algorithm) {
	case "JRip": // nominal-only
	    JRip jr = new JRip();
	    jr.buildClassifier(dataset);
	    // FastVector fv = jr.getRuleset();
	    // for (int i = 0; i < jr.getRuleset().size(); i++) {
	    // LOGGER.info(((RipperRule)
	    // (fv.elementAt(i))).toString(dataset.classAttribute()));
	    // }
	    /************************* 10-fold Cross-validation ************************/
	    int folds = 10;
	    int seed = 1;
	    Random rand = new Random(seed);
	    Evaluation evalJ = new Evaluation(dataset);
	    evalJ.crossValidateModel(jr, dataset, folds, rand);
	    /******************************************************************/
	    //
	    // ArrayList<Rule> fv = jr.getRuleset();
	    // // JSONArray ja = new JSONArray();
	    //
	    // for (int i = 0; i < fv.size(); i++) {
	    // // ja.put(((RipperRule)
	    // (fv.elementAt(i))).toString(dataset.classAttribute()));
	    // }
	    // // jsonResponse.put("Rules", ja);
	    // LOGGER.info("ErrorRate", eval.errorRate());
	    // LOGGER.info("Precision", eval.precision(0));
	    // LOGGER.info("Recall", eval.recall(0));
	    // LOGGER.info("fMeasure", eval.fMeasure(0));
	    //
	    // }
	    LOGGER.info(evalJ.toSummaryString("Eval:\n", false));
	    break;
	case "Regression":
	    DataSource sourceTest = new DataSource(testFile);
	    Instances datasetTest = sourceTest.getDataSet();
	    datasetTest.setClassIndex(datasetTest.numAttributes() - 1);

	    LinearRegression lr = new LinearRegression();
	    lr.buildClassifier(dataset);
	    LOGGER.info("" + lr);
	    Evaluation evalR = new Evaluation(dataset);
	    evalR.evaluateModel(lr, datasetTest);
	    LOGGER.info(evalR.toSummaryString("Eval:\n", false));

	    DataSource sourcePredict = new DataSource(predictFile);
	    Instances datasetPredict = sourcePredict.getDataSet();
	    datasetPredict.setClassIndex(datasetPredict.numAttributes() - 1);

	    for (int i = 0; i < datasetPredict.numInstances(); i++) {
		double actualVal = datasetPredict.instance(i).classValue();
		Instance newInst = datasetPredict.instance(i);
		double predictedVal = lr.classifyInstance(newInst);
		LOGGER.info(Math.round(actualVal) + " predicted -> " + Math.round(predictedVal));
	    }
	    modelFileName = dataName + "_model.model";
	    weka.core.SerializationHelper.write(modelFileName, lr);

	    break;
	case "Ibk":
	    DataSource sourceTestIbK = new DataSource(testFile);
	    Instances datasetTestIbk = sourceTestIbK.getDataSet();
	    datasetTestIbk.setClassIndex(datasetTestIbk.numAttributes() - 1);

	    IBk ibk = new IBk(10);
	    ibk.buildClassifier(dataset);
	    LOGGER.info("" + ibk);
	    Evaluation evalIbk = new Evaluation(dataset);
	    evalIbk.evaluateModel(ibk, datasetTestIbk);
	    LOGGER.info(evalIbk.toSummaryString("Eval:\n", false));

	    DataSource sourcePredictIbK = new DataSource(predictFile);
	    Instances datasetPredictIbk = sourcePredictIbK.getDataSet();
	    datasetPredictIbk.setClassIndex(datasetPredictIbk.numAttributes() - 1);

	    for (int i = 0; i < datasetPredictIbk.numInstances(); i++) {
		double actualVal = datasetPredictIbk.instance(i).classValue();
		Instance newInst = datasetPredictIbk.instance(i);
		double predictedVal = ibk.classifyInstance(newInst);

		LOGGER.info(actualVal + " predicted -> " + predictedVal);
	    }
	    break;
	case "RbfNet": // very bad performance for predicting next car position
	    DataSource sourceTestRbfNet = new DataSource(testFile);
	    Instances datasetTestRbfNet = sourceTestRbfNet.getDataSet();
	    datasetTestRbfNet.setClassIndex(datasetTestRbfNet.numAttributes() - 1);

	    RBFNetwork rbfNet = new RBFNetwork();
	    rbfNet.buildClassifier(dataset);
	    LOGGER.info("" + rbfNet);
	    Evaluation evalRbfNet = new Evaluation(datasetTestRbfNet);
	    evalRbfNet.evaluateModel(rbfNet, datasetTestRbfNet);
	    LOGGER.info(evalRbfNet.toSummaryString("Eval:\n", false));

	    DataSource sourcePredictRbfNet = new DataSource(predictFile);
	    Instances datasetPredictRbfNet = sourcePredictRbfNet.getDataSet();
	    datasetPredictRbfNet.setClassIndex(datasetPredictRbfNet.numAttributes() - 1);

	    for (int i = 0; i < datasetPredictRbfNet.numInstances(); i++) {
		double actualVal = datasetPredictRbfNet.instance(i).classValue();
		Instance newInst = datasetPredictRbfNet.instance(i);
		double predictedVal = rbfNet.classifyInstance(newInst);
		LOGGER.info(actualVal + " predicted -> " + predictedVal);
	    }
	    break;
	case "LogR": // nominal-only
	    DataSource sourceTestLogR = new DataSource(testFile);
	    Instances datasetTestLogR = sourceTestLogR.getDataSet();
	    datasetTestLogR.setClassIndex(datasetTestLogR.numAttributes() - 1);

	    Logistic logR = new Logistic();
	    logR.buildClassifier(dataset);
	    LOGGER.info("" + logR);
	    Evaluation evalLogR = new Evaluation(datasetTestLogR);
	    evalLogR.evaluateModel(logR, datasetTestLogR);
	    LOGGER.info(evalLogR.toSummaryString("Eval:\n", false));

	    DataSource sourcePredictLogR = new DataSource(predictFile);
	    Instances datasetPredictLogR = sourcePredictLogR.getDataSet();
	    datasetPredictLogR.setClassIndex(datasetPredictLogR.numAttributes() - 1);

	    for (int i = 0; i < datasetPredictLogR.numInstances(); i++) {
		double actualVal = datasetPredictLogR.instance(i).classValue();
		Instance newInst = datasetPredictLogR.instance(i);
		double predictedVal = logR.classifyInstance(newInst);
		LOGGER.info(actualVal + " predicted -> " + predictedVal);
	    }
	    break;
	}
	return modelFileName;
    }

    public String calculatePredictions(String algorithm, String dataName, int numberOfPredictions) throws Exception {
	String predictedValues = "";
	switch (algorithm) {
	case "JRip": // nominal-only
	    break;
	case "Regression":

	    LinearRegression lr = (LinearRegression) weka.core.SerializationHelper.read(dataName + "_model.model");

	    for (int i = 0; i < numberOfPredictions; i++) {
		DataSource sourcePredict = new DataSource(dataName + "_predict.arff");
		Instances datasetPredict = sourcePredict.getDataSet();
		datasetPredict.setClassIndex(datasetPredict.numAttributes() - 1);
		// double actualVal = datasetPredict.instance(i).classValue();
		Instance newInst = datasetPredict.instance(i);
		double predictedVal = lr.classifyInstance(newInst);
		LOGGER.info("predicted -> " + Math.round(predictedVal));
		// LOGGER.info(Math.round(actualVal) + " predicted -> " +
		// Math.round(predictedVal));
		List<String> newLineInstance = new ArrayList<>();
		for (int j = 1; j < datasetPredict.numAttributes(); j++) {
		    newLineInstance.add(String.valueOf(Math.round(newInst.value(j))));
		}
		newLineInstance.add(String.valueOf(Math.round(predictedVal)));
		predictedValues += Math.round(predictedVal) + "\n";
		try {
		    File file = new File(dataName + "_predict.arff");
		    if (!file.exists()) {
			file.createNewFile();
		    }

		    FileWriter fileWritter = new FileWriter(file, true);
		    BufferedWriter output = new BufferedWriter(fileWritter);
		    output.write(newLineInstance.toString().substring(1, newLineInstance.toString().length() - 1)
			    .replace(" ", "") + "\n");
		    output.close();
		} catch (IOException e) {
		    e.printStackTrace();
		}
	    }
	    break;
	case "Ibk":
	    break;
	case "RbfNet": // very bad performance for predicting next car position
	    break;
	case "LogR": // nominal-only
	    break;
	}
	return predictedValues;
    }
}
