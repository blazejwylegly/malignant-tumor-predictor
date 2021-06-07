package pl.polsl.tumorpredictor.data;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class DataEntry {
    /** 1-5 scale assessment - ordinal */
    private Double bi_rads;

    /** Patient's age in year */
    private Double age;

    /**
    * Categorical mass shape
    * round=1
    * oval=2
    * lobular=3
    * irregular=4
    * */
    private Double shape;

    /**
     * Categorical mass margin
     * circumscribed=1
     * microlobulated=2
     * obscured=3
     * ill-defined=4
     * spiculated=5
     * */
    private Double massMargin;

    /**
     * Mass density - ordinal
     * high=1
     * iso=2
     * low=3
     * fat-containing=4
     */
    private Double massDensity;

    /**
     * 0/1 value that indicates the severity
     * */
    private Double severity;
}
