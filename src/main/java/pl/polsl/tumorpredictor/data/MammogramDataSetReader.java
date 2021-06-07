package pl.polsl.tumorpredictor.data;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.ui.standalone.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Component
@NoArgsConstructor
@AllArgsConstructor
public class MammogramDataSetReader {

    private RecordReader recordReader;

    private static final int numClasses = 2;
    private static final int BI_RADS_INDEX = 0;
    private static final int AGE_INDEX = 1;
    private static final int SHAPE_INDEX = 2;
    private static final int MARGIN_INDEX = 3;
    private static final int DENSITY_INDEX = 4;
    private static final int SEVERITY_INDEX = 5;

    public MammogramDataSet readDatasetFile(String filename)
            throws IOException, InterruptedException {
        initializeReader(filename);
        List<List<Writable>> dataSet = retrieveCsvDataset();
        return createMammogramDataSet(dataSet);
    }

    private MammogramDataSet createMammogramDataSet(List<List<Writable>> dataSet) {
        List<DataEntry> dataEntries = dataSet.stream()
                .map(this::mapSingleSliceToDataEntry)
                .collect(Collectors.toList());
        return MammogramDataSet.of(dataEntries);
    }

    private DataEntry mapSingleSliceToDataEntry(List<Writable> singleRecord) {
        return DataEntry.builder()
                .bi_rads(getDoubleOrNull(singleRecord, BI_RADS_INDEX))
                .age(getDoubleOrNull(singleRecord, AGE_INDEX))
                .shape(getDoubleOrNull(singleRecord, SHAPE_INDEX))
                .massMargin(getDoubleOrNull(singleRecord, MARGIN_INDEX))
                .massDensity(getDoubleOrNull(singleRecord, DENSITY_INDEX))
                .severity(getDoubleOrNull(singleRecord, SEVERITY_INDEX))
                .build();
    }

    private Double getDoubleOrNull(List<Writable> singleRecord, int index) {
        try {
            return singleRecord.get(index).toDouble();
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    private List<List<Writable>> retrieveCsvDataset() {
        List<List<Writable>> records = new ArrayList<>();
        while(recordReader.hasNext()) {
            records.add(recordReader.next());
        }
        return records;
    }

    private void initializeReader(String datasetFileName) throws IOException, InterruptedException {
        recordReader = new CSVRecordReader(1, ",");
        recordReader.initialize(new FileSplit(
                new ClassPathResource(datasetFileName).getFile()));
    }
}
