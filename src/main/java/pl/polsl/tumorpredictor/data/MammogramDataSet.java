package pl.polsl.tumorpredictor.data;

import lombok.AllArgsConstructor;

import java.util.List;
import java.util.function.Function;

@AllArgsConstructor(staticName = "of")
public class MammogramDataSet {
    private List<DataEntry> dataEntries;

    public Double[] getFeatureArray(Function<? super DataEntry, Double> mapper) {
        return dataEntries.stream()
                .map(mapper)
                .toArray(Double[]::new);
    }
}
