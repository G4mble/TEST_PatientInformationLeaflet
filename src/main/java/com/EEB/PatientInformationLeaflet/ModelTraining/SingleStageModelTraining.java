package com.EEB.PatientInformationLeaflet.ModelTraining;

import com.EEB.PatientInformationLeaflet.ModelUsage.ModelOutputTest;
import com.EEB.Preprocessing.StringPreprocessor;
import com.EEB.Tokenizer.GermanNGramTokenizerFactory;
import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class SingleStageModelTraining
{
    private static Logger _log = LoggerFactory.getLogger(ModelOutputTest.class);

    //TODO use small dataset
//    private static final String _filename = "text_input/news.en.heldout-00000-of-00050";
    private static final String _filename = "text_input/test.txt";

    //TODO use large dataset
//    private static final String _filename = "text_input/news.en-00001-of-00100";

    public static void main(String[] args) throws Exception
    {
        long startTime = System.currentTimeMillis();
//        _log.info("Attempting to load dataset...");
//        File dataset = new File(new ClassPathResource(_filename).getFile().getAbsolutePath());
//        if (!dataset.exists())
//        {
//            _log.error("Dataset does not exist at: " + _filename);
//            _log.info("Terminating...");
//            return;
//        }

        _log.info("Configuring input parameters...");
//        SentenceIterator sentenceIterator = new BasicLineIterator(dataset);
        SentenceIterator sentenceIterator = new FileSentenceIterator(new File(new ClassPathResource("text_input").getFile().getAbsolutePath()));
//        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        TokenizerFactory defaultTokenizerFactory = new DefaultTokenizerFactory();
        TokenizerFactory tokenizerFactory = new GermanNGramTokenizerFactory(defaultTokenizerFactory, 1, 3);
        tokenizerFactory.setTokenPreProcessor(new StringPreprocessor());

        _log.info("Building model...");
        //TODO TS evaluate current && further parameters
        Word2Vec model = new Word2Vec.Builder()
//                .useHierarchicSoftmax(false)
//                .negativeSample(10)
                .minWordFrequency(1)
                .allowParallelTokenization(false)
                .workers(8)
                //how often one batch size is iterated
                .iterations(1)
                //how often the whole corpus [and thereby each batch/iteration] is processed
                .epochs(1)
                //                .batchSize(1000)
                //size of the feature vector
                .layerSize(200)
                //initialization seed for the feature vector [keep the same for reproducible results]
                .seed(34)
                //context window size
                .windowSize(6)
                .iterate(sentenceIterator)
                .tokenizerFactory(tokenizerFactory)
                .stopWords(StopWords.getStopWords())
                .build();

        _log.info("Start model training...");
        model.fit();
        _log.info("Model training completed.");
        _log.info("Elapsed time since start: " + (System.currentTimeMillis() - startTime) + " ms");

        _log.info("Testing model:");

        // vocabulary output
        FileUtils.writeLines(new File("vocabulary.txt"), model.getVocab().words());

        System.out.println("Testoutput: " + model.hasWord("the"));
        System.out.println("Testoutput: " + model.hasWord("when"));
        System.out.println("Testoutput: " + model.hasWord("then"));
        System.out.println("Testoutput: " + model.hasWord("and"));
        System.out.println("Testoutput: " + model.hasWord("now"));
        System.out.println("Testoutput: " + model.hasWord("criminal"));
        System.out.println("Testoutput: " + model.hasWord(" criminal"));
        System.out.println("Testoutput: " + model.hasWord("color"));
        System.out.println("Testoutput: " + model.wordsNearest("color", 10));

//        _log.info("Saving model...");
//        DateFormat dateFormat = new SimpleDateFormat("YYYY-MM-dd_hh-mm-ss");
//        WordVectorSerializer.writeWord2VecModel(model, "model_output_" + dateFormat.format(new Date()) + "_test.cdf");
    }
}