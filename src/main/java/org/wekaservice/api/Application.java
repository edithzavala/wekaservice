package org.wekaservice.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.wekaservice.logic.WekaMiner;

@SpringBootApplication
public class Application {

    public static WekaMiner WEKA_MINER;
    public static final String DATA_LOCATION = "/tmp/weka/";

    public static void main(String[] args) {
	SpringApplication.run(Application.class, args);
	WEKA_MINER = new WekaMiner();
    }

}
