/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package mnist;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class MnistFeedAutoEncoder {

    public static final String MODEL = "C:\\temp\\autoencoder.model";
    private static Logger log = LoggerFactory.getLogger(MnistFeedAutoEncoder.class);

    public static void main(String[] args) throws Exception {
        new MnistFeedAutoEncoder().run();
    }

    protected void run() throws Exception {
        MultiLayerNetwork model = createNet();
        model.init();

        DataSetIterator mnistTrain = new MnistDataSetIterator(128, true, 12345);
        trainNet(model, mnistTrain);
        //model = ModelSerializer.restoreMultiLayerNetwork(new File(MODEL));

        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 12345);
        evaluateModel(model, mnistTest);
    }

    private void evaluateModel(MultiLayerNetwork model, DataSetIterator mnistTest) {
        log.info("Evaluate model....");
        System.setProperty("java.specification.version", "1.8");

        for (int i = 0; i < 10; i++) {
            INDArray input = Nd4j.zeros(new int[]{1, 10});
            input.putScalar(new int[]{0, i}, 10f);
            INDArray output = model.activateSelectedLayers(2, 3, input);
            saveImage("Output_" + i, output);
        }
        log.info("****************Evaluate finished********************");
    }

    private void saveImage(String name, INDArray output) {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                image.getRaster().setSample(y, x, 0, 255 * output.getDouble(x * 28 + y));
            }
        }
        try {
            ImageIO.write(image, "PNG", new File("c:\\temp\\" + name + ".png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void trainNet(MultiLayerNetwork model, DataSetIterator mnistTrain) throws Exception {
        log.info("Train model....");
        //TODO train with default mnist iterator, but output should be = input
        // model.save(new File(MODEL));
    }


    private MultiLayerNetwork createNet() {
        log.info("Build model....");
        MultiLayerConfiguration conf = null;
        //TODO define enets
        return new MultiLayerNetwork(conf);
    }

    private void createUiServer(MultiLayerNetwork model) {
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        int listenerFrequency = 1;
        model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
    }
}
