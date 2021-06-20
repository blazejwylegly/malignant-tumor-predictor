package pl.polsl.tumorpredictor.reporting;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Builder
@Getter
class PartialReportResult {
    @Setter
    private String generationStart;
    private String accuracy;
    private String precision;
    private String recall;
    private String f1_score;

    @Override
    public String toString() {
        return String.format("%s,%s,%s,%s,%s",
                generationStart,
                accuracy,
                precision,
                recall,
                f1_score
        );
    }
}