package pl.polsl.tumorpredictor.network;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NeuralNetBuilder {
    private static final int NUM_INPUTS = 5;
    private static final int NUM_CLASSES = 2;
    private static final long SEED = 8;

    public static MultiLayerNetwork createNetwork() {
        var net = new MultiLayerNetwork(prepareNetworkConfiguration());
        net.init();
        return net;
    }

    private static MultiLayerConfiguration prepareNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(NUM_INPUTS).nOut(7)
                        .build())
                .layer(new DenseLayer.Builder().nIn(7).nOut(6)
                        .build())
//                .layer(new DenseLayer.Builder().nIn(10).nOut(6)
//                        .build())
//                .layer(new DenseLayer.Builder().nIn(6).nOut(6)
//                        .build())
                .layer(new DenseLayer.Builder().nIn(6).nOut(4)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(4).nOut(NUM_CLASSES).build())
                .build();
    }
}
