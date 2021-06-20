package pl.polsl.tumorpredictor.reporting;

import lombok.Builder;
import lombok.Getter;

@Builder
@Getter
class ReportHeader {
    private String mainActivationFunction;
    private String weighInitFunction;
    private int numLayers;
    private String outputLayerActivationFunction;
    private int networkSeed;

    @Override
    public String toString() {
        return "mainActivationFunction='" + mainActivationFunction + '\'' +
                ", weighInitFunction='" + weighInitFunction + '\'' +
                ", numLayers=" + numLayers +
                ", outputLayerActivationFunction='" + outputLayerActivationFunction + '\'' +
                ", networkSeed=" + networkSeed;
    }
}
