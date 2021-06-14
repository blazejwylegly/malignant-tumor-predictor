package pl.polsl.tumorpredictor.data;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.tuple.Tuples;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;

@Component
@NoArgsConstructor
@AllArgsConstructor
@Slf4j
public class MammogramDataSetReader {

    private RecordReader recordReader;

    private static final int NUM_LINES_TO_SKIP = 0;
    private static final char DELIMITER = ',';

    private static final int LABEL_INDEX = 5;
    private static final int NUM_CLASSES = 2;
    private static final int BATCH_SIZE = 829;

    public Pair<DataSet, DataSet> readDatasetFile(String filename, double ratio)
            throws IOException, InterruptedException {
        if((ratio >= 0) && (ratio <= 1)) {
            initializeReader(filename);
            return retrieveShuffledDataSet(ratio);
        } else {
            throw new RuntimeException("Invalid split ratio!");
        }
    }

    private Pair<DataSet, DataSet> retrieveShuffledDataSet(double ratio) {
        DataSetIterator iterator = new RecordReaderDataSetIterator(
                recordReader, BATCH_SIZE, LABEL_INDEX, NUM_CLASSES
        );
        DataSet dataSet = iterator.next();
        dataSet.shuffle();
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(ratio);

        DataSet trainingDataset = testAndTrain.getTrain();
        log.info("Successfully populated training data set with {} entries", trainingDataset.asList().size());
        DataSet testingDataset = testAndTrain.getTest();
        log.info("Successfully populated testing data set with {} entries", testingDataset.asList().size());

        return Tuples.pair(trainingDataset, testingDataset);
    }

    private void initializeReader(String datasetFileName) throws IOException, InterruptedException {
        recordReader = new CSVRecordReader(NUM_LINES_TO_SKIP, DELIMITER);
        recordReader.initialize(new FileSplit(new File(datasetFileName)));
    }
}
