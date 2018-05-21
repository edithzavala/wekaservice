package org.wekaservice.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class WekaServiceApi {

    @GetMapping("/{dataName}/{algorithmId}")
    public ResponseEntity<String> getModel(@PathVariable("dataName") String dataName,
	    @PathVariable("algorithmId") String algorithmId) {
	String modelFileName = "";
	try {
	    modelFileName = Application.WEKA_MINER.buildModel(Application.DATA_LOCATION + dataName, algorithmId);
	} catch (Exception e) {
	    e.printStackTrace();
	}
	return ResponseEntity.ok(modelFileName);

    }

    @GetMapping("/{dataName}/{algorithmId}/{numberOfPredictions}")
    public ResponseEntity<String> getPrediction(@PathVariable("dataName") String dataName,
	    @PathVariable("algorithmId") String algorithmId,
	    @PathVariable("numberOfPredictions") int numberOfPredictions) {
	String predictions = "";
	try {
	    predictions = Application.WEKA_MINER.calculatePredictions(algorithmId, Application.DATA_LOCATION + dataName,
		    numberOfPredictions);
	} catch (Exception e) {
	    e.printStackTrace();
	}
	return ResponseEntity.ok(predictions);

    }
}
