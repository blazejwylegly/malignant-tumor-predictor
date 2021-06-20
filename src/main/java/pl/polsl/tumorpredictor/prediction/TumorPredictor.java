package pl.polsl.tumorpredictor.prediction;

import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import pl.polsl.tumorpredictor.data.MammogramDataSetReader;
import pl.polsl.tumorpredictor.network.NeuralNetMeta;
import pl.polsl.tumorpredictor.network.NeuralNetBuilder;
import pl.polsl.tumorpredictor.reporting.ReportingTool;

import java.io.IOException;

@Slf4j
@NoArgsConstructor
@Component
public class TumorPredictor {
    private static final String TRAINING_DATASET_CSV = "src/main/resources/dataset_training.csv";
    private static final String TESTING_DATASET_CSV = "src/main/resources/dataset_testing.csv";

    private MammogramDataSetReader dataSetRetriever;
    private DataSet trainingDataSet;
    private DataSet testingDataSet;

    private MultiLayerNetwork network;
    private NeuralNetMeta config;
    private ReportingTool reportingTool;

    @Autowired
    public TumorPredictor(MammogramDataSetReader dataSetRetriever,
                          NeuralNetMeta config,
                          ReportingTool reportingTool) {
        this.dataSetRetriever = dataSetRetriever;
        this.config = config;
        this.reportingTool = reportingTool;
    }

    public void predict() throws IOException {
        setup();
        trainNetworkAndCollectResults();
    }

    private void trainNetworkAndCollectResults() throws IOException {
        for (int i = 1; i <= 250000; i++) {
            network.fit(trainingDataSet);
            if (i % 1000 == 0) {
                log.info("Current iteration " + i);
                reportingTool.generatePartialReport(
                        network, trainingDataSet, i
                );
            }
        }
        reportingTool.flush("results.csv");
    }

    private void setup() {
        setupDataSets();
        setupNetwork();
    }

    private void setupDataSets() {
        tryToInitializeDataSet();
        normalizeDataSets();
    }

    private void tryToInitializeDataSet() {
        try {
            this.trainingDataSet = dataSetRetriever.readDatasetFile(TRAINING_DATASET_CSV);
            this.testingDataSet = dataSetRetriever.readDatasetFile(TESTING_DATASET_CSV);
        } catch (IOException | InterruptedException e) {
            log.error("Error occurred while initializing data set! Terminating application...");
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void normalizeDataSets() {
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingDataSet);           //Collect the statistics from the training data. This does not modify the input data
        normalizer.transform(trainingDataSet);     //Apply normalization to the training data

        normalizer.transform(testingDataSet);      //Apply normalization to the test data. This is using statistics calculated from the *training* set
    }

    private void setupNetwork() {
        this.network = NeuralNetBuilder.createNetwork(this.config);
    }
}
