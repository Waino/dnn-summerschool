import numpy

from lm import train

trainerr, validerr, testerr = train(train_path='../data/morph/finnish.clean.train10k',
                                    validation_path='../data/morph/finnish.clean.test',
                                    dictionary_path='../data/morph/morph.vocab',
                                    model_path='/l/senarvi/theano-rnnlm/model-train10k-lstm.npz',
                                    reload_state=True,
                                    dim_word=256,
                                    dim=1024,
                                    n_words=30000,
                                    decay_c=0.,
                                    lrate=0.0001,
                                    optimizer='adam', 
                                    maxlen=30,
                                    batch_size=32,
                                    valid_batch_size=16,
                                    validFreq=5000,
                                    dispFreq=20,
                                    saveFreq=40,
                                    sampleFreq=20)
print(validerr)

