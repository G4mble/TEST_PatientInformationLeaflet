package com.EEB.PatientInformationLeaflet.ModelUsage;

import com.EEB.Preprocessing.GermanStem;
import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

public class ModelOutputTest
{
    private static Logger _log = LoggerFactory.getLogger(ModelOutputTest.class);

    //TODO change filename --> you can put your models in the "reources/model_output" folder
    private static final String _filename = "model_output/M_002_model_output_2018-05-31_12-56-12.cmf";

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

        String aspirinStem = GermanStem.stem("aspirin");
        String ibuStem = GermanStem.stem("ibuprofen");
        String kopfschmerzStem = GermanStem.stem("kopfschmerzen");
        String nebenwStem = GermanStem.stem("nebenwirkung");
        String anwendungStem = GermanStem.stem("anwendung");
        String medStem = GermanStem.stem("medikament");
        String schmerzmittelStem = GermanStem.stem("schmerzmittel");

//        String aspirinStem = "aspirin";
//        String ibuStem = "ibuprofen";
//        String kopfschmerzStem = "kopfschmerzen";
//        String nebenwStem = "nebenwirkung";
//        String anwendungStem = "anwendung";
//        String medStem = "medikament";
//        String schmerzmittelStem = "schmerzmittel";

        StringBuilder builder = new StringBuilder();
        builder.append("HasWord(aspirin): ").append(word2VecModel.hasWord(aspirinStem));
        builder.append("\n\n");
        builder.append("HasWord(ibuprofen): ").append(word2VecModel.hasWord(ibuStem));
        builder.append("\n\n");
        builder.append("HasWord(kopfschmerzen): ").append(word2VecModel.hasWord(kopfschmerzStem));
        builder.append("\n\n");
        builder.append("HasWord(nebenwirkung): ").append(word2VecModel.hasWord(nebenwStem));
        builder.append("\n\n");
        builder.append("HasWord(anwendung): ").append(word2VecModel.hasWord(anwendungStem));
        builder.append("\n\n");
        builder.append("HasWord(medikament): ").append(word2VecModel.hasWord(medStem));
        builder.append("\n\n");
        builder.append("HasWord(schmerzmittel): ").append(word2VecModel.hasWord(schmerzmittelStem));
        builder.append("\n\n");
        builder.append("similarWordsTo(aspirin): ").append(word2VecModel.similarWordsInVocabTo(aspirinStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(ibuprofen): ").append(word2VecModel.similarWordsInVocabTo(ibuStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(kopfschmerzen): ").append(word2VecModel.similarWordsInVocabTo(kopfschmerzStem, .90d));
        builder.append("\n\n");
        builder.append("wordsNearest(aspirin): ").append(word2VecModel.wordsNearest(aspirinStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearestSUM(aspirin): ").append(word2VecModel.wordsNearestSum(aspirinStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(ibuprofen): ").append(word2VecModel.wordsNearest(ibuStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearestSUM(ibuprofen): ").append(word2VecModel.wordsNearestSum(ibuStem, 10));
        builder.append("\n\n");
        builder.append("similarity(aspirin, ibuprofen): ").append(word2VecModel.similarity(aspirinStem, ibuStem));
        builder.append("\n\n");
        builder.append("similarity(aspirin, kopfschmerzen): ").append(word2VecModel.similarity(aspirinStem, kopfschmerzStem));
        builder.append("\n\n");
        builder.append("similarity(kopfschmerzen, ibuprofen): ").append(word2VecModel.similarity(kopfschmerzStem, ibuStem));
        builder.append("\n\n");
        builder.append("wordsNearest(aspirin+Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(aspirinStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearestSUM(aspirin+Nebenwirkung): ").append(word2VecModel.wordsNearestSum(word2VecModel.getWordVectorsMean(Arrays.asList(aspirinStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(ibuprofen+Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(ibuStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearestSUM(ibuprofen+Nebenwirkung): ").append(word2VecModel.wordsNearestSum(word2VecModel.getWordVectorsMean(Arrays.asList(ibuStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(Kopfschmerz+Anwendung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(kopfschmerzStem, anwendungStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearestSUM(Kopfschmerz+Anwendung): ").append(word2VecModel.wordsNearestSum(word2VecModel.getWordVectorsMean(Arrays.asList(kopfschmerzStem, anwendungStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(kopfschmerzen, schmerzmittel): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(kopfschmerzStem, anwendungStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearestSUM(kopfschmerzen, schmerzmittel): ").append(word2VecModel.wordsNearestSum(word2VecModel.getWordVectorsMean(Arrays.asList(kopfschmerzStem, anwendungStem)), 10));
        builder.append("\n\n");
        builder.append("Similarity(aspirin, medikament): ").append(word2VecModel.similarity(aspirinStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(ibuprofen, medikament): ").append(word2VecModel.similarity(ibuStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(kopfschmerzen, nebenwirkung): ").append(word2VecModel.similarity(kopfschmerzStem, nebenwStem));
        builder.append("\n\n");
        builder.append("NEW wordsNearest(aspirin, nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorMatrix(aspirinStem).add(word2VecModel.getWordVectorMatrix(nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("NEW wordsNearest(ibuprofen, nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorMatrix(ibuStem).add(word2VecModel.getWordVectorMatrix(nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("NEW wordsNearest(kopfschmerzen, schmerzmittel): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorMatrix(kopfschmerzStem).add(word2VecModel.getWordVectorMatrix(schmerzmittelStem)), 10));

        FileUtils.write(new File("modelEvaluation.txt"), builder.toString());
    }
}