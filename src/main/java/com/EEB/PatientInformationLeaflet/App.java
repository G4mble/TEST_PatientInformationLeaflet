package com.EEB.PatientInformationLeaflet;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;

public class App
{
    private static Logger _log = LoggerFactory.getLogger(App.class);

    //TODO change filename --> you can put your models in the "reources/model_output" folder
    private static final String _filename = "model_output/model_output_FULL.txt";

    public static void main( String[] args )
    {
        File modelFile = new File(_filename);
        if(!modelFile.exists())
        {
            _log.error("Model file does not exist at: " + _filename);
            _log.info("Terminating...");
            return;
        }

        _log.info("Attempting to load Word2Vec Model from file: \"" + _filename + "\" ...");
        Word2Vec word2VecModel = WordVectorSerializer.readWord2VecModel(modelFile);
        _log.info("Loading complete.");

        //TODO do something with the model
        _log.info("SomeResult: " + word2VecModel.hasWord("word"));
        _log.info("SomeResult: " + word2VecModel.similarity("word1", "word2"));
        _log.info("SomeResult: " + word2VecModel.similarWordsInVocabTo("word1", .5d));
        _log.info("SomeResult: " + word2VecModel.wordsNearest("word", 5));
        _log.info("SomeResult: " + word2VecModel.wordsNearestSum("word", 5));
        _log.info("SomeResult: " + word2VecModel.accuracy(Arrays.asList("word")));
    }
}