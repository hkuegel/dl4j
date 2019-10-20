package lstm;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

public class LSTMGenerator {

    private static final int SEQ_LEN = 32;
    private static final String LSTM_MODEL = "C:\\temp\\lstm.model";
    private static final String SHAKESPEAREMODEL = "C:\\temp\\shakespearemodel\\";
    private static final int BATCH_SIZE = 100;
    private static Logger log = LoggerFactory.getLogger(LSTMGenerator.class);
    private CharIterator charIterator;

    public LSTMGenerator() {
        charIterator = new CharIterator("C:\\temp\\shakespeare.txt", SEQ_LEN, BATCH_SIZE);
    }

    public static void main(String[] args) throws Exception {
        new LSTMGenerator().run(true, false, 3000, "Es war einmal");
    }

    private void run(boolean train, boolean restore, int samplesize, String initializer) throws Exception {
        MultiLayerNetwork model = restore ? restoreModel() : createModel();
        if (train) {
            trainNet(model, charIterator);
            model.save(new File(LSTM_MODEL));
        }
        System.out.println(generateText(model, samplesize, initializer));
    }

    private MultiLayerNetwork restoreModel() throws Exception {
        // or load from checkpoint
        //String modelFile = CheckpointListener.lastCheckpoint(new File(SHAKESPEAREMODEL)).getFilename();
        //MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(SHAKESPEAREMODEL + modelFile));
        return ModelSerializer.restoreMultiLayerNetwork(new File(LSTM_MODEL));
    }

    private String generateText(MultiLayerNetwork model, int sampleSize, String initializer) {

        Map<Integer, Integer> charToIdx = charIterator.getCharToIdx();
        Map<Integer, Integer> idxToChar = charToIdx.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
        warmUpNet(model, initializer, charToIdx);

        StringBuilder sampleText = sampleText(model, sampleSize, idxToChar, charToIdx);
        System.out.println(sampleText.toString());

        return sampleText.toString();
    }

    private StringBuilder sampleText(MultiLayerNetwork model, int sampleSize, Map<Integer, Integer> idxToChar,
                                     Map<Integer, Integer> charToIdx) {
        Random r = new Random(34352442);
        StringBuilder sampleText = new StringBuilder();
        INDArray input = getInput(' ', idxToChar);
        INDArray output;
        //model.rnnClearPreviousState();
        for (int s = 0; s < sampleSize; s++) {
            output = model.rnnTimeStep(input);
            int nextChar = sampleCharBasedOnOutputProb(output, r, idxToChar);
            input = getInput((char) nextChar, charToIdx);
            sampleText.append((char) nextChar);
        }
        return sampleText;
    }


    private void warmUpNet(MultiLayerNetwork model, String initializer, Map<Integer, Integer> charToIdx) {
        INDArray output = null;
        for (char c : initializer.toCharArray()) {
            INDArray input = getInput(c, charToIdx);
            model.rnnTimeStep(input);
        }
    }


    private INDArray getInput(char c, Map<Integer, Integer> charToIdx) {
        INDArray input = Nd4j.zeros(new int[]{1, charToIdx.size()});
        int oneHot = charToIdx.get((int) c);
        input.putScalar(new int[]{0, oneHot}, 1f);
        return input;
    }

    private int sampleCharBasedOnOutputProb(INDArray output, Random r, Map<Integer, Integer> indexToChar) {
        double randomProb = r.nextDouble();
        double probSum = 0;
        for (int i = 0; i < indexToChar.size(); i++) {
            double probOfI = output.getDouble(new int[]{0, i});
            probSum += probOfI;
            if (probSum > randomProb) {
                return indexToChar.get(i);
            }
        }
        return indexToChar.get(indexToChar.size() - 1);
    }


    private void trainNet(MultiLayerNetwork model, DataSetIterator charIterator) {
        log.info("Train model....");
        //createUiServer(model);
        //CheckpointListener checkPoint = new CheckpointListener.Builder(new File("C:\\temp\\shakespearemodel2\\")).saveEveryNIterations(1000).build();
        model.setListeners(new ComposableIterationListener(new ScoreIterationListener(10)));
        for (int i = 0; i < 10000; i++) {
            model.fit(charIterator.next());
            if (i % 10 == 0) {
                generateText(model, 100, "");
            }
        }
    }


    private MultiLayerNetwork createModel() {
        int numberOfCharClasses = charIterator.getCharToIdx().size();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.005))
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new LSTM.Builder().nIn(numberOfCharClasses).nOut(512)
                        .activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nIn(512).nOut(512)
                        .activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(512).nOut(numberOfCharClasses).build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTLength(50)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }


}
