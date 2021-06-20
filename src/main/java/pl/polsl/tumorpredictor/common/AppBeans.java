package pl.polsl.tumorpredictor.common;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import pl.polsl.tumorpredictor.network.NeuralNetMeta;

@Configuration
public class AppBeans {

    public static final int NUM_INPUTS = 5;
    public static final int NUM_CLASSES = 2;

    @Bean
    public static NeuralNetMeta networkConfig() {
        Activation outputLayerActivation = Activation.SOFTMAX;

        return NeuralNetMeta.builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .outputLayerActivation(outputLayerActivation)
                .layers(new Layer[]{
                        new DenseLayer.Builder().nIn(NUM_INPUTS).nOut(7).build(),
                        new DenseLayer.Builder().nIn(7).nOut(6).build(),
                        new DenseLayer.Builder().nIn(6).nOut(4).build(),
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(outputLayerActivation)
                                .nIn(4).nOut(NUM_CLASSES).build()
                })
                .networkSeed(8)
                .build();
    }

}
