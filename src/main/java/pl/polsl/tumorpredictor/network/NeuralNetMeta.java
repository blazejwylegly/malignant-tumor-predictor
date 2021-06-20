package pl.polsl.tumorpredictor.network;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

@Getter
@Builder
public class NeuralNetMeta {

    private Activation activation;
    private WeightInit weightInit;
    private Layer[] layers;
    private Activation outputLayerActivation;
    private int networkSeed;
}
