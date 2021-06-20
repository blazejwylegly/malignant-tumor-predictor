package pl.polsl.tumorpredictor.reporting;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import pl.polsl.tumorpredictor.network.NeuralNetMeta;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

@Component
@Slf4j
public class ReportingTool {

    private ReportHeader reportHeader;
    private List<PartialReportResult> reportResults;
    private NeuralNetMeta netMeta;

    @Autowired
    public ReportingTool(NeuralNetMeta netMeta) {
        this.netMeta = netMeta;
        prepareHeader();
        reportResults = new ArrayList<>();
    }

    private void prepareHeader() {
        this.reportHeader = ReportHeader.builder()
                .mainActivationFunction(netMeta.getActivation().name())
                .numLayers(netMeta.getLayers().length)
                .outputLayerActivationFunction(netMeta.getOutputLayerActivation().name())
                .networkSeed(netMeta.getNetworkSeed())
                .weighInitFunction(netMeta.getWeightInit().name())
                .build();
    }

    public void generatePartialReport(MultiLayerNetwork network,
                                      DataSet dataSet,
                                      int iteration) {
        Evaluation eval = new Evaluation(2);
        INDArray output = network.output(dataSet.getFeatures());
        eval.eval(dataSet.getLabels(), output);
        addPartialReport(eval.stats(), iteration);
    }

    private void addPartialReport(String stats, int iteration) {
        PartialReportResult result = EvaluationStatisticsParser.parsePartialStatistics(stats);
        result.setGenerationStart(Integer.toString(iteration));
        reportResults.add(result);
    }

    public void flush(String reportFileName) throws IOException {
        try(PrintWriter pw = new PrintWriter(reportFileName)) {
            pw.println(reportHeader.toString());
            reportResults
                    .forEach(pw::println);
        }
    }
}
