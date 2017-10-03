from primitiv import DefaultScopeDevice
from primitiv import DefaultScopeGraph
from primitiv import Device
from primitiv import CPUDevice
from primitiv import Graph
from primitiv import Node
from primitiv import Parameter
from primitiv.trainers import SGD
from primitiv import Shape
from primitiv.initializers import Constant
from primitiv.initializers import XavierUniform
from primitiv import operators as F

import random
import sys
import numpy as np


NUM_TRAIN_SAMPLES = 60000
NUM_TEST_SAMPLES = 10000
NUM_INPUT_UNITS = 28 * 28
NUM_HIDDEN_UNITS = 800
NUM_OUTPUT_UNITS = 10
BATCH_SIZE = 200
NUM_TRAIN_BATCHES = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
NUM_TEST_BATCHES = int(NUM_TEST_SAMPLES / BATCH_SIZE)
MAX_EPOCH = 100


def load_images(filename, n):
    try:
        ifs = open(filename, "rb")
    except:
        print("File could not be opened:", filename, file=sys.stderr)
        sys.exit(1)
    ifs.seek(16)
    ret = (np.fromfile(ifs, dtype=np.uint8, count=n*NUM_INPUT_UNITS) / 255).astype(np.float32).reshape((n, NUM_INPUT_UNITS))
    ifs.close()
    return ret


# Helper function to load labels.
def load_labels(filename, n):
    try:
        ifs = open(filename, "rb")
    except:
        print("File could not be opened:", filename, file=sys.stderr)
        sys.exit(1)
    ifs.seek(8)  # header
    return np.fromfile(ifs, dtype=np.uint8, count=n).astype(np.uint32)


def main():
    # Loads data
    train_inputs = load_images("data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES)
    train_labels = load_labels("data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES)
    test_inputs = load_images("data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES)
    test_labels = load_labels("data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES)

    # Uses GPU.
    #dev = CUDADevice(0)
    with DefaultScopeDevice(CPUDevice()):

        # Parameters for the multilayer perceptron.
        pw1 = Parameter("w1", [NUM_HIDDEN_UNITS, NUM_INPUT_UNITS], XavierUniform())
        pb1 = Parameter("b1", [NUM_HIDDEN_UNITS], Constant(0))
        pw2 = Parameter("w2", [NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS], XavierUniform())
        pb2 = Parameter("b2", [NUM_OUTPUT_UNITS], Constant(0))

        # Parameters for batch normalization.
        #Parameter pbeta("beta", {NUM_HIDDEN_UNITS}, Constant(0));
        #Parameter pgamma("gamma", {NUM_HIDDEN_UNITS}, Constant(1));

        # Trainer
        trainer = SGD(.5)
        trainer.add_parameter(pw1)
        trainer.add_parameter(pb1)
        trainer.add_parameter(pw2)
        trainer.add_parameter(pb2)
        #trainer.add_parameter(&pbeta);
        #trainer.add_parameter(&pgamma);

        # Helper lambda to construct the predictor network.
        def make_graph(inputs, train):
            # Stores input values.
            x = F.input(data=inputs)
            # Calculates the hidden layer.
            w1 = F.input(param=pw1)
            b1 = F.input(param=pb1)
            h = F.relu(F.matmul(w1, x) + b1)
            # Batch normalization
            #Node beta = F::input(pbeta);
            #Node gamma = F::input(pgamma);
            #h = F::batch::normalize(h) * gamma + beta;
            # Dropout
            h = F.dropout(h, .5, train)
            # Calculates the output layer.
            w2 = F.input(param=pw2)
            b2 = F.input(param=pb2)
            return F.matmul(w2, h) + b2

        ids = list(range(NUM_TRAIN_SAMPLES))

        for epoch in range(MAX_EPOCH):
            # Shuffles sample IDs.
            random.shuffle(ids)

            # Training loop
            for batch in range(NUM_TRAIN_BATCHES):
                print("\rTraining... %d / %d" % (batch + 1, NUM_TRAIN_BATCHES), end="")
                inputs = train_inputs[ids[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]]
                labels = train_labels[ids[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]]

                trainer.reset_gradients()

                # Constructs the graph.
                g = Graph()
                with DefaultScopeGraph(g):
                    y = make_graph(inputs, True)
                    loss = F.softmax_cross_entropy(y, labels, 0)
                    avg_loss = F.batch.mean(loss)

                    # Dump computation graph at the first time.
                    #if (epoch == 0 && batch == 0) g.dump();

                    # Forward, backward, and updates parameters.
                    g.forward(avg_loss)
                    g.backward(avg_loss)

                    trainer.update()

            print()

            match = 0

            # Test loop
            for batch in range(NUM_TEST_BATCHES):
                print("\rTesting... %d / %d" % (batch + 1, NUM_TEST_BATCHES), end="")
                # Makes a test minibatch.
                inputs = test_inputs[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]

                # Constructs the graph.
                with Graph() as g:
                    y = make_graph(inputs, False)

                    # Gets outputs, argmax, and compares them with the label.
                    y_val = g.forward(y).to_list()
                    for i in range(BATCH_SIZE):
                        maxval = -1e10
                        argmax = -1
                        for j in range(NUM_OUTPUT_UNITS):
                            v = y_val[j + i * NUM_OUTPUT_UNITS]
                            if (v > maxval):
                                maxval = v
                                argmax = j
                        if argmax == test_labels[i + batch * BATCH_SIZE]:
                            match += 1

            accuracy = 100.0 * match / NUM_TEST_SAMPLES
            print("\nepoch %d: accuracy: %.2f%%\n" % (epoch, accuracy))
            #pw1.save("mnist-params-w1.param");
            #pb1.save("mnist-params-b1.param");
            #pw2.save("mnist-params-w2.param");
            #pb2.save("mnist-params-b2.param");
            #pbeta.save("mnist-params-beta.param");
            #pgamma.save("mnist-params-gamma.param");
            #cout << "epoch " << epoch << ": saved parameters." << endl;


if __name__ == "__main__":
    main()
