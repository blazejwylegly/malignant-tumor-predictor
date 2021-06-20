package pl.polsl.tumorpredictor.network;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.learning.config.Sgd;

public class NeuralNetBuilder {

    public static MultiLayerNetwork createNetwork(NeuralNetMeta neuralNetMeta) {
        var net = new MultiLayerNetwork(
                prepareNetworkConfiguration(neuralNetMeta)
        );
        net.init();
        return net;
    }

    private static MultiLayerConfiguration prepareNetworkConfiguration(NeuralNetMeta neuralNetMeta) {
        return new NeuralNetConfiguration.Builder()
                .seed(neuralNetMeta.getNetworkSeed())
                .activation(neuralNetMeta.getActivation())
                .weightInit(neuralNetMeta.getWeightInit())
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list(neuralNetMeta.getLayers())
                .build();
    }
}
