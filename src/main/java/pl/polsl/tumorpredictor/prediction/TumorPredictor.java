package pl.polsl.tumorpredictor.prediction;

import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.eclipse.collections.api.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import pl.polsl.tumorpredictor.data.MammogramDataSetReader;
import pl.polsl.tumorpredictor.network.NeuralNetBuilder;

import java.io.IOException;

@Slf4j
@NoArgsConstructor
@Component
public class TumorPredictor {
    private static final String DATASET_FILENAME = "src/main/resources/mammographic_masses.csv";

    private MammogramDataSetReader dataSetRetriever;
    private DataSet trainingDataSet;
    private DataSet testingDataSet;

    private MultiLayerNetwork network;

    @Autowired
    public TumorPredictor(MammogramDataSetReader dataSetRetriever) {
        this.dataSetRetriever = dataSetRetriever;
    }

    public void predict() {
        setup();
        trainNetwork();
        evaluateNetwork();
    }

    private void trainNetwork() {
        network.setListeners(new ScoreIterationListener(50));
        for (int i = 0; i < 20000; i++) {
            network.fit(trainingDataSet);
        }
    }

    private void evaluateNetwork() {
        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = network.output(testingDataSet.getFeatures());
        eval.eval(testingDataSet.getLabels(), output);
        log.info(eval.stats());
    }

    private void setup() {
        tryToInitializeDataSet();
        normalizeDataSets();
        this.network = NeuralNetBuilder.createNetwork();
    }

    private void normalizeDataSets() {
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingDataSet);           //Collect the statistics from the training data. This does not modify the input data
        normalizer.transform(trainingDataSet);     //Apply normalization to the training data

        normalizer.transform(testingDataSet);      //Apply normalization to the test data. This is using statistics calculated from the *training* set
    }

    private void tryToInitializeDataSet() {

        try {
            Pair<DataSet, DataSet> dataSets =
                    dataSetRetriever.readDatasetFile(DATASET_FILENAME, 0.7);
            this.trainingDataSet = dataSets.getOne();
            this.testingDataSet = dataSets.getTwo();
        } catch (IOException | InterruptedException e) {
            log.error("Error occurred while initializing data set! Terminating application...");
            e.printStackTrace();
            System.exit(1);
        }
    }
}
