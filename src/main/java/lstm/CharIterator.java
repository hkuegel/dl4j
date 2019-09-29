package lstm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class CharIterator implements DataSetIterator {

    private final int[] text;
    private final Map<Integer, Integer> charToIdx;
    private int pos = 0;
    private int seqLen;
    private int batchSize;
    private Random random;

    CharIterator(String fileName, int seqLen, int batchSize) {
        random = new Random(123456789);
        this.seqLen = seqLen;
        this.batchSize = batchSize;
        File f = new File(fileName);
        text = readTextInput(f);
        charToIdx = createIndex(text);
        System.out.println("Read text completed, text size = " + text.length);
        System.out.println("Different chats = " + charToIdx.size());

    }

    private Map<Integer, Integer> createIndex(int[] text) {
        Map<Integer, Integer> idxMap = new HashMap<>();
        for (int value : text) {
            if (!idxMap.containsKey(value)) {
                idxMap.put(value, idxMap.size());
            }
        }
        return idxMap;
    }


    private int[] readTextInput(File f) {
        try {
            int[] text = new int[(int) f.length()];
            int c, i = 0;
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(f), Charset.forName("UTF-8")));
            while ((c = reader.read()) != -1) {
                text[i++] = c;
            }
            return text;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public DataSet next(int batchSize) {
        pos = random.nextInt(text.length - seqLen - 1);

        System.out.println("Generating sample at : " + pos);
        // [miniBatchSize,inputSize,timeSeriesLength]
        INDArray features = Nd4j.zeros(batchSize, charToIdx.size(), seqLen);
        INDArray labels = Nd4j.zeros(batchSize, charToIdx.size(), seqLen);
        //TODO implement
        // masks : many-to-many, but ignore first half of output (warm up)
        // all input columns, but only final output is relevant
        INDArray featureMask = Nd4j.zeros(new int[]{batchSize, seqLen});
        INDArray labelMask = Nd4j.zeros(new int[]{batchSize, seqLen});

        pos += batchSize;
        return new DataSet(features, labels, featureMask, labelMask);
    }

    @Override
    public int inputColumns() {
        return charToIdx.size();
    }

    @Override
    public int totalOutcomes() {
        return charToIdx.size();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        pos = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return true;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    Map<Integer, Integer> getCharToIdx() {
        return charToIdx;
    }
}
