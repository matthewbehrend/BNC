from adaptation_methods.adapt_BNC import BNC
import mxnet as mx
from mxnet import nd, gluon

class BNCInspect(BNC):
    # BatchNorm Classifier
    # network segmented to allow bypassing the BN layer

    def __init__(self, k_classes, context, bypassbn=False):
        super().__init__(k_classes, context)
        self.bypassbn = bypassbn

        # Build classification network
        self.classifierNet  = self.classifierHead()
        self.bnNet          = self.addBatchNorm( gluon.nn.Sequential() )
        self.classifierNet.initialize(mx.init.Normal(0.02), ctx=self.context)
        self.bnNet.initialize(mx.init.Normal(0.02), ctx=self.context)

    def initTrainer(self, data_sample):
        self.model(data_sample)
        self.classifier_trainer  = gluon.Trainer(self.classifierNet.collect_params(), 'adam', {'learning_rate': self.base_learning_rate})
        self.bn_trainer          = gluon.Trainer(self.bnNet.collect_params(), 'adam', {'learning_rate': self.base_learning_rate})

    def stepTrainer(self):
        self.classifier_trainer.step(1)
        if(not self.bypassbn):
            self.bn_trainer.step(1)
        
    def log_softmax(self, x):
        return nd.log_softmax(x, axis=1)

    def model(self, data):
        if(self.bypassbn):
            return self.log_softmax(self.classifierNet(data))
        else:
            return self.log_softmax(self.bnNet( self.classifierNet(data) ))

    def get_softmax_input_all_data(self, data):
        if(self.bypassbn):
            softmax_input = self.classifierNet(data)
        else:
            softmax_input = self.bnNet( self.classifierNet(data) )
        return softmax_input.asnumpy()

    def get_fc_output_all_data(self, data):
        fc_output = self.classifierNet(data)
        return fc_output.asnumpy()
    
    def get_fc_tensor_all_data(self, data):
        # print(self.classifierNet.collect_params())
        layer_keys = self.classifierNet.collect_params().keys()
        L1_weight_key = [a for a in layer_keys if '_dense1_weight' in a]
        if(len(L1_weight_key) != 1):
            return None
        return self.classifierNet.collect_params()[ L1_weight_key[0] ].data().asnumpy().flatten()


    