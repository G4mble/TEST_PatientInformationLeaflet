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

public class ModelOutputTest
{
    private static Logger _log = LoggerFactory.getLogger(ModelOutputTest.class);

    //TODO change filename --> you can put your models in the "reources/model_output" folder
    private static final String _filename = "model_output/MW_005_model_output_2018-06-03_12-09-16.cmf";

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

//        String aspirinStem = GermanStem.stem("aspirin");
//        String ibuStem = GermanStem.stem("ibuprofen");
//        String kopfschmerzStem = GermanStem.stem("kopfschmerzen");
//        String nebenwStem = GermanStem.stem("nebenwirkung");
//        String anwendungStem = GermanStem.stem("anwendung");
//        String medStem = GermanStem.stem("medikament");
//        String schmerzmittelStem = GermanStem.stem("schmerzmittel");
//        String kontraIndStem = GermanStem.stem("kontraindikationen");
//        String vivimedStem = GermanStem.stem("vivimed");
//        String temaginStem = GermanStem.stem("temagin");
//        String zentragressStem = GermanStem.stem("zentragress");
//        String retortapyrinStem = GermanStem.stem("retortapyrin");

//        String koenigStem = GermanStem.stem("koenig");
//        String koeniginStem = GermanStem.stem("koenigin");
//        String mannStem = GermanStem.stem("mann");
//        String frauStem = GermanStem.stem("frau");

        String koenigStem = "koenig";
        String koeniginStem = "koenigin";
        String mannStem = "mann";
        String frauStem = "frau";

        String aspirinStem = "aspirin";
        String ibuStem = "ibuprofen";
        String kopfschmerzStem = "kopfschmerzen";
        String nebenwStem = "nebenwirkung";
        String anwendungStem = "anwendung";
        String medStem = "medikament";
        String schmerzmittelStem = "schmerzmittel";
        String kontraIndStem = "kontraindikationen";
        String vivimedStem = "vivimed";
        String temaginStem = "temagin";
        String zentragressStem = "zentragress";
        String retortapyrinStem = "retortapyrin";

        StringBuilder builder = new StringBuilder();

        builder.append("HasWord(koenig): ").append(word2VecModel.hasWord(koenigStem));
        builder.append("\n\n");
        builder.append("HasWord(koenigin): ").append(word2VecModel.hasWord(koeniginStem));
        builder.append("\n\n");
        builder.append("HasWord(mann): ").append(word2VecModel.hasWord(mannStem));
        builder.append("\n\n");
        builder.append("HasWord(frau): ").append(word2VecModel.hasWord(frauStem));
        builder.append("\n\n");
        builder.append("similarity(koenig, koenigin): ").append(word2VecModel.similarity(koenigStem, koeniginStem));
        builder.append("\n\n");
        builder.append("similarity(mann, frau): ").append(word2VecModel.similarity(mannStem, frauStem));
        builder.append("\n\n");
        builder.append("similarity(mann, koenig): ").append(word2VecModel.similarity(mannStem, koenigStem));
        builder.append("\n\n");
        builder.append("similarity(frau, koenigin): ").append(word2VecModel.similarity(frauStem, koeniginStem));
        builder.append("\n\n");
        builder.append("wordsNearest(koenig-mann+frau): ").append(word2VecModel.wordsNearest(Arrays.asList(koenigStem, frauStem), Arrays.asList(mannStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(koenigin-frau+mann): ").append(word2VecModel.wordsNearest(Arrays.asList(koeniginStem, mannStem), Arrays.asList(frauStem), 10));
        builder.append("\n\n");
        builder.append("---------------------------------------------------------------");
        builder.append("---------------------------------------------------------------");
        builder.append("\n\n");



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
        builder.append("HasWord(kontraindikationen): ").append(word2VecModel.hasWord(kontraIndStem));
        builder.append("\n\n");
        builder.append("HasWord(vivimed): ").append(word2VecModel.hasWord(vivimedStem));
        builder.append("\n\n");
        builder.append("HasWord(temagin): ").append(word2VecModel.hasWord(temaginStem));
        builder.append("\n\n");
        builder.append("HasWord(zentragress): ").append(word2VecModel.hasWord(zentragressStem));
        builder.append("\n\n");
        builder.append("HasWord(retortapyrin): ").append(word2VecModel.hasWord(retortapyrinStem));
        builder.append("\n\n");

        builder.append("similarWordsTo(aspirin): ").append(word2VecModel.similarWordsInVocabTo(aspirinStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(ibuprofen): ").append(word2VecModel.similarWordsInVocabTo(ibuStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(kopfschmerzen): ").append(word2VecModel.similarWordsInVocabTo(kopfschmerzStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(medikament): ").append(word2VecModel.similarWordsInVocabTo(medStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(vivimed): ").append(word2VecModel.similarWordsInVocabTo(vivimedStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(temagin): ").append(word2VecModel.similarWordsInVocabTo(temaginStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(zentragress): ").append(word2VecModel.similarWordsInVocabTo(zentragressStem, .90d));
        builder.append("\n\n");
        builder.append("similarWordsTo(retortapyrin): ").append(word2VecModel.similarWordsInVocabTo(retortapyrinStem, .90d));
        builder.append("\n\n");


        builder.append("similarity(nebenwirkungen, kontraindikationen): ").append(word2VecModel.similarity(nebenwStem, kontraIndStem));
        builder.append("\n\n");
        builder.append("similarity(aspirin, ibuprofen): ").append(word2VecModel.similarity(aspirinStem, ibuStem));
        builder.append("\n\n");


        builder.append("similarity(aspirin, vidimed): ").append(word2VecModel.similarity(vivimedStem, ibuStem));
        builder.append("\n\n");
        builder.append("similarity(aspirin, temagin): ").append(word2VecModel.similarity(temaginStem, ibuStem));
        builder.append("\n\n");
        builder.append("similarity(aspirin, zentragress): ").append(word2VecModel.similarity(zentragressStem, ibuStem));
        builder.append("\n\n");
        builder.append("similarity(aspirin, retortapyrin): ").append(word2VecModel.similarity(retortapyrinStem, ibuStem));
        builder.append("\n\n");
        builder.append("similarity(vivimed, temagin): ").append(word2VecModel.similarity(vivimedStem, temaginStem));
        builder.append("\n\n");
        builder.append("similarity(vivimed, zentragress): ").append(word2VecModel.similarity(vivimedStem, zentragressStem));
        builder.append("\n\n");
        builder.append("similarity(vivimed, retortapyrin): ").append(word2VecModel.similarity(vivimedStem, retortapyrinStem));
        builder.append("\n\n");
        builder.append("similarity(temagin, zentragress): ").append(word2VecModel.similarity(temaginStem, zentragressStem));
        builder.append("\n\n");
        builder.append("similarity(temagin, retortapyrin): ").append(word2VecModel.similarity(temaginStem, retortapyrinStem));
        builder.append("\n\n");
        builder.append("similarity(zentragress, retortapyrin): ").append(word2VecModel.similarity(zentragressStem, retortapyrinStem));
        builder.append("\n\n");
        builder.append("Similarity(ibuprofen, vivimed): ").append(word2VecModel.similarity(ibuStem, vivimedStem));
        builder.append("\n\n");
        builder.append("Similarity(ibuprofen, temagin): ").append(word2VecModel.similarity(ibuStem, temaginStem));
        builder.append("\n\n");
        builder.append("Similarity(ibuprofen, zentragress): ").append(word2VecModel.similarity(ibuStem, zentragressStem));
        builder.append("\n\n");
        builder.append("Similarity(ibuprofen, retortapyrin): ").append(word2VecModel.similarity(ibuStem, retortapyrinStem));
        builder.append("\n\n");
        builder.append("similarity(aspirin, kopfschmerzen): ").append(word2VecModel.similarity(aspirinStem, kopfschmerzStem));
        builder.append("\n\n");
        builder.append("similarity(kopfschmerzen, ibuprofen): ").append(word2VecModel.similarity(kopfschmerzStem, ibuStem));
        builder.append("\n\n");
        builder.append("similarity(kopfschmerzen, vivimed): ").append(word2VecModel.similarity(kopfschmerzStem, vivimedStem));
        builder.append("\n\n");
        builder.append("similarity(kopfschmerzen, temagin): ").append(word2VecModel.similarity(kopfschmerzStem, temaginStem));
        builder.append("\n\n");
        builder.append("similarity(kopfschmerzen, zentragress): ").append(word2VecModel.similarity(kopfschmerzStem, zentragressStem));
        builder.append("\n\n");
        builder.append("similarity(kopfschmerzen, retortapyrin): ").append(word2VecModel.similarity(kopfschmerzStem, retortapyrinStem));
        builder.append("\n\n");
        builder.append("Similarity(aspirin, medikament): ").append(word2VecModel.similarity(aspirinStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(ibuprofen, medikament): ").append(word2VecModel.similarity(ibuStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(vivimed, medikament): ").append(word2VecModel.similarity(vivimedStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(temagin, medikament): ").append(word2VecModel.similarity(temaginStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(zentragress, medikament): ").append(word2VecModel.similarity(zentragressStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(retortapyrin, medikament): ").append(word2VecModel.similarity(retortapyrinStem, medStem));
        builder.append("\n\n");
        builder.append("Similarity(kopfschmerzen, nebenwirkung): ").append(word2VecModel.similarity(kopfschmerzStem, nebenwStem));
        builder.append("\n\n");

        builder.append("wordsNearest(aspirin): ").append(word2VecModel.wordsNearest(aspirinStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(kontraindikationen): ").append(word2VecModel.wordsNearest(kontraIndStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(ibuprofen): ").append(word2VecModel.wordsNearest(ibuStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(vivimed): ").append(word2VecModel.wordsNearest(vivimedStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(temagin): ").append(word2VecModel.wordsNearest(temaginStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(zentragress): ").append(word2VecModel.wordsNearest(zentragressStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(retortapyrin): ").append(word2VecModel.wordsNearest(retortapyrinStem, 10));
        builder.append("\n\n");
        builder.append("wordsNearest(vivimed (+) kontraindikationen): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(vivimedStem, kontraIndStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(vivimed (+) Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(vivimedStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(temagin (+) kontraindikationen): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(temaginStem, kontraIndStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(temagin (+) Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(temaginStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(zentragress (+) kontraindikationen): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(zentragressStem, kontraIndStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(zentragress (+) Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(zentragressStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(retortapyrin (+) kontraindikationen): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(retortapyrinStem, kontraIndStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(retortapyrin (+) Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(retortapyrinStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(aspirin (+) kontraindikationen): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(aspirinStem, kontraIndStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(aspirin (+) Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(aspirinStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(ibuprofen (+) kontraindikationen): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(ibuStem, kontraIndStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(ibuprofen (+) Nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(ibuStem, nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(Kopfschmerz (+) Anwendung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(kopfschmerzStem, anwendungStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(kopfschmerzen (+) schmerzmittel): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorsMean(Arrays.asList(kopfschmerzStem, schmerzmittelStem)), 10));
        builder.append("\n\n");
        builder.append("NEW wordsNearest(aspirin (+) nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorMatrix(aspirinStem).add(word2VecModel.getWordVectorMatrix(nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("NEW wordsNearest(ibuprofen (+) nebenwirkung): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorMatrix(ibuStem).add(word2VecModel.getWordVectorMatrix(nebenwStem)), 10));
        builder.append("\n\n");
        builder.append("NEW wordsNearest(kopfschmerzen (+) schmerzmittel): ").append(word2VecModel.wordsNearest(word2VecModel.getWordVectorMatrix(kopfschmerzStem).add(word2VecModel.getWordVectorMatrix(schmerzmittelStem)), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(ibuprofen (-) kopfschmerzen): ").append(word2VecModel.wordsNearest(Arrays.asList(ibuStem), Arrays.asList(kopfschmerzStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(aspirin (-) kopfschmerzen): ").append(word2VecModel.wordsNearest(Arrays.asList(aspirinStem), Arrays.asList(kopfschmerzStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(vivimed (-) kopfschmerzen): ").append(word2VecModel.wordsNearest(Arrays.asList(vivimedStem), Arrays.asList(kopfschmerzStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(temagin (-) kopfschmerzen): ").append(word2VecModel.wordsNearest(Arrays.asList(temaginStem), Arrays.asList(kopfschmerzStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(zentragress (-) kopfschmerzen): ").append(word2VecModel.wordsNearest(Arrays.asList(zentragressStem), Arrays.asList(kopfschmerzStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(retortapyrin (-) kopfschmerzen): ").append(word2VecModel.wordsNearest(Arrays.asList(retortapyrinStem), Arrays.asList(kopfschmerzStem), 10));
        builder.append("\n\n");
        builder.append("wordsNearest(medikament)").append(word2VecModel.wordsNearest(medStem, 20));

        FileUtils.write(new File("modelEvaluation.txt"), builder.toString());
    }
}