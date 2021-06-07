package pl.polsl.tumorpredictor;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import pl.polsl.tumorpredictor.prediction.TumorPredictor;

import java.io.IOException;

@SpringBootApplication
public class TumorPredictorApplication implements CommandLineRunner {

    @Autowired
    private TumorPredictor tumorPredictor;

    public static void main(String[] args) throws IOException {
        SpringApplication.run(TumorPredictorApplication.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        tumorPredictor.predict();
    }
}
