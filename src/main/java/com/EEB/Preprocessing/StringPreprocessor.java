package com.EEB.Preprocessing;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

import java.util.regex.Pattern;

public class StringPreprocessor implements TokenPreProcess
{

    private final Pattern _preprocessPattern = Pattern.compile("[\\d!\"§$%&/()=?`ß´²³{\\[\\]}\\\\+*~#'\\-_.:,;<>|^°@€\\uFFFD]+");
    private final Pattern _nonAsciiPattern = Pattern.compile("[^\\x00-\\x7F]");

    @Override
    public String preProcess(String token)
    {
        String output = token.toLowerCase();
        output = _preprocessPattern.matcher(output).replaceAll("");
        output = _nonAsciiPattern.matcher(output).replaceAll("");
        return output.length() > 1 ? output.trim() : "";
    }
}