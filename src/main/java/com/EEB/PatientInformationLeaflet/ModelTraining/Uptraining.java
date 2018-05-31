package com.EEB.PatientInformationLeaflet.ModelTraining;

import org.apache.commons.io.FileUtils;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Date;
import java.util.stream.Stream;

/**
 * This is simple example for model weights update after initial vocab building.
 * If you have built your w2v model, and some time later you've decided that it can be
 * additionally trained over new corpus, here's an example how to do it.
 *
 * PLEASE NOTE: At this moment, no new words will be added to vocabulary/model.
 * Only weights update process will be issued. It's often called "frozen vocab training".
 *
 * @author raver119@gmail.com
 */
public class Uptraining {

    private static Logger log = LoggerFactory.getLogger(Uptraining.class);

    public static void main(String[] args)
    {
        Word2Vec vecInternal = new Word2Vec.Builder()
                .minWordFrequency(20)
                .allowParallelTokenization(false)
                .workers(8)
                .iterations(1)
                .epochs(1)
                .layerSize(50)
                .seed(34)
                .windowSize(6)
                .stopWords(StopWords.getStopWords())
                .build();

        try (Stream<Path> paths = Files.walk(Paths.get("C:\\Users\\Tommy\\Documents\\IntelliJIdea\\Projects\\Word2Vec_PatientInformationLeaflet\\src\\main\\resources\\data\\wikidump\\_AA")))
        {
            paths.filter(Files::isRegularFile).forEach(x -> processFile(x, vecInternal));
            DateFormat dateFormat = new SimpleDateFormat("YYYY-MM-dd_hh-mm-ss");
            FileUtils.writeLines(new File("vocabulary_" + dateFormat.format(new Date()) + ".txt"), vecInternal.getVocab().words());
        }
        catch (Exception e)
        {
            System.out.println("ERROR!!");
        }
    }

    private static void processFile(Path path, Word2Vec vecInternal)
    {
        try
        {
            SentenceIterator iterator = new BasicLineIterator(path.toFile());
            TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
            tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

            vecInternal.setTokenizerFactory(tokenizerFactory);
            vecInternal.setSentenceIterator(iterator);

            vecInternal.fit();
        }
        catch (FileNotFoundException e)
        {
            System.out.println("Second Error");
        }
    }
}

