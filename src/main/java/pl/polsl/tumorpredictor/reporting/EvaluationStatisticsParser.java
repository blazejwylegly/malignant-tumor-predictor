package pl.polsl.tumorpredictor.reporting;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class EvaluationStatisticsParser {
    private static final Pattern ACCURACY_PATTERN = Pattern.compile("(A.*)(0,\\d{4})");
    private static final Pattern PRECISION_PATTERN = Pattern.compile("(P.*)(0,\\d{4})");
    private static final Pattern RECALL_PATTERN = Pattern.compile("(R.*)(0,\\d{4})");
    private static final Pattern F1_SCORE_PATTERN = Pattern.compile("(F.*)(0,\\d{4})");

    static PartialReportResult parsePartialStatistics(String statistics) {

        String accuracy = matchAndExtractPattern(ACCURACY_PATTERN, statistics);
        String precision = matchAndExtractPattern(PRECISION_PATTERN, statistics);
        String recall = matchAndExtractPattern(RECALL_PATTERN, statistics);
        String f1_score = matchAndExtractPattern(F1_SCORE_PATTERN, statistics);

        return PartialReportResult.builder()
                .accuracy(csvFormat(accuracy))
                .precision(csvFormat(precision))
                .recall(csvFormat(recall))
                .f1_score(csvFormat(f1_score))
                .build();
    }

    private static String csvFormat(String input) {
        return input.replaceAll(",", ".");
    }

    private static String matchAndExtractPattern(Pattern pattern, String input) {
        Matcher matcher = pattern.matcher(input);
        if (matcher.find()) {
            return matcher.group(2);
        }
        return null;
    }
}
