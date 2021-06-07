package pl.polsl.tumorpredictor.prediction;

import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import pl.polsl.tumorpredictor.data.DataEntry;
import pl.polsl.tumorpredictor.data.MammogramDataSet;
import pl.polsl.tumorpredictor.data.MammogramDataSetReader;

import java.io.IOException;
import java.util.Arrays;

@Slf4j
@NoArgsConstructor
@Component
public class TumorPredictor {
    private static final String DATASET_FILENAME = "mammographic_masses.csv";

    private static final int BATCH_SIZE = 961;
    private static final int LABEL_INDEX = 6;
    private static final int NUM_CLASSES = 2;

    private MammogramDataSetReader dataSetRetriever;
    private MammogramDataSet dataSet;

    @Autowired
    public TumorPredictor(MammogramDataSetReader dataSetRetriever) {
        this.dataSetRetriever = dataSetRetriever;
    }

    public void predict() {
        tryToInitializeDataSet();
        printDataSet(dataSet);
    }

    private void tryToInitializeDataSet() {

         try {
             dataSet = dataSetRetriever.readDatasetFile(DATASET_FILENAME);
         } catch (IOException | InterruptedException e) {
            log.error("Error occurred while initializing data set! Terminating application...");
            e.printStackTrace();
            System.exit(1);
         }
    }

    private static void printDataSet(MammogramDataSet dataSet) {
        Double[] ages = dataSet.getFeatureArray(DataEntry::getAge);
        Arrays.stream(ages).forEach(System.out::println);
    }
}
