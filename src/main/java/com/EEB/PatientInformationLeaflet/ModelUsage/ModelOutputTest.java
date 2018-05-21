package com.EEB.PatientInformationLeaflet.ModelUsage;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class ModelOutputTest
{
    private static Logger _log = LoggerFactory.getLogger(ModelOutputTest.class);

    //TODO change filename --> you can put your models in the "reources/model_output" folder
    private static final String _filename = "model_output/model_output_FULL.txt";

    public static void main( String[] args ) throws Exception
    {
        File modelFile = new File(new ClassPathResource(_filename).getFile().getAbsolutePath());
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
//        _log.info("SomeResult: " + word2VecModel.hasWord("word"));
//        _log.info("SomeResult: " + word2VecModel.similarity("word1", "word2"));
//        _log.info("SomeResult: " + word2VecModel.similarWordsInVocabTo("word1", .5d));
//        _log.info("SomeResult: " + word2VecModel.wordsNearest("word", 5));
//        _log.info("SomeResult: " + word2VecModel.wordsNearestSum("word", 5));
//        _log.info("SomeResult: " + word2VecModel.accuracy(Arrays.asList("word")));

        System.out.println("TestOutput: " + word2VecModel.hasWord("dax"));
        System.out.println("TestOutput: " + word2VecModel.hasWord("nikkei"));
        System.out.println("TestOutput: " + word2VecModel.similarWordsInVocabTo("dax", 0.8d));
        System.out.println("TestOutput: " + word2VecModel.similarWordsInVocabTo("nikkei", 0.8d));
        System.out.println("TestOutput: " + word2VecModel.wordsNearest("dax", 10));
        System.out.println("TestOutput: " + word2VecModel.wordsNearest("nikkei", 10));
        System.out.println("TestOutput: " + word2VecModel.similarity("dax", "nikkei"));
    }
}