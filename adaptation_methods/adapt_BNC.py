from adaptation_methods.base_model import BaseEncoderClass, TrainerTraceLog
import mxnet as mx
from mxnet import nd, gluon, autograd


class BNC(BaseEncoderClass):
    # BatchNorm Classifier (BNC)
    # source-free unsupervised domain adaptation method
    # trains on source domain then switches to adaptation on target domain, without access to source data
    
    def __init__(self, k_classes, context):
        super().__init__(k_classes, context)
        # Build classification network
        self.bnClassNet = self.classifierHead()
        self.bnClassNet = self.addBatchNorm(self.bnClassNet)
        self.bnClassNet = self.addSoftmax(self.bnClassNet)
        self.bnClassNet.initialize(mx.init.Normal(0.02), ctx=self.context)
    
    def initTrainer(self, data_sample):
        self.model(data_sample)
        self.bnclass_trainer    = gluon.Trainer(self.bnClassNet.collect_params(), 'adam', {'learning_rate': self.base_learning_rate})
                
    def stepTrainer(self):
        self.bnclass_trainer.step(1)

    def trainOptimization(self, data_loader, n_epochs, domainTests):
        # Supervised in Source Domain
        tracelog = TrainerTraceLog()
        cumulative_points = 0
            
        for e in range(n_epochs):
            epoch_loss = 0
            for i, (data, label) in enumerate(data_loader['source']): # Source domain
                data        = data.as_in_context( self.context )
                label_hot   = nd.one_hot(label.as_in_context( self.context ), depth=self.k_classes)
                n_points    = data.shape[0]
                
                with autograd.record():
                    logP_c  = self.model(data)
                    loss        = nd.mean( self.crossEntropy(label_hot, logP_c) )

                    loss.backward()
                self.stepTrainer()

                epoch_loss          += loss.asscalar()
                cumulative_points   += n_points
                
            # log at completion of epoch
            tracelog.logPerformanceData(self, domainTests, e, cumulative_points, epoch_loss/(1+i))
        return tracelog
    
    def adaptOptimization(self, data_loader, n_epochs, domainTests):
        # Unsupervised in Target Domain
        tracelog = TrainerTraceLog()
        cumulative_points = 0
            
        for e in range(n_epochs):
            epoch_loss = 0
            for i, (tgt_data, _) in enumerate(data_loader['target']): # Target domain, without labels
                tgt_data = tgt_data.as_in_context( self.context )
                n_points = tgt_data.shape[0]
                
                with autograd.record():
                    logP_c = self.model( tgt_data )
                    loss        = nd.mean( self.crossEntropy(nd.exp(logP_c), logP_c) ) # entropy
                    
                    loss.backward()
                self.stepTrainer()
                
                epoch_loss          += loss.asscalar()
                cumulative_points   += n_points

            # log at completion of epoch
            tracelog.logPerformanceData(self, domainTests, e, cumulative_points, epoch_loss/(1+i))
        return tracelog

    def trainSequence(self, source_data_loader, target_data_loader, epochs=5):
        # source training
        adapt_data_loaders, domainTests = self.initForTraining(source_data_loader, target_data_loader)
        trace_log = self.trainOptimization(adapt_data_loaders, epochs, domainTests)
        trace_log.epoch = [-1 for x in trace_log.epoch] # relabel epochs as -1 before adaptation

        # target adaptation
        adapt_data_loaders, domainTests = self.initForTraining(source_data_loader, target_data_loader)
        trace_log2 = self.adaptOptimization(adapt_data_loaders, epochs, domainTests)

        trace_log.append(trace_log2)
        return trace_log
        
    def model(self, data):
        return self.bnClassNet(data)


class BNC_Cotrained(BNC):
    # co-trained on source and target domain data
    # this class is just for comparison of performance if source data can be used during adaptation
    
    def __init__(self, k_classes, context):
        super().__init__(k_classes, context)
    
    def trainOptimization(self, data_loader, n_epochs, domainTests):    
        tracelog = TrainerTraceLog()
        cumulative_points = 0
            
        for e in range(n_epochs):
            epoch_loss = 0
            for i, (data, label) in enumerate(data_loader['source']):
                data    =  data.as_in_context( self.context )
                label   = label.as_in_context( self.context )
                label_hot   = nd.one_hot(label, depth=self.k_classes)
                n_points = data.shape[0]
                
                for (tgt_data, _) in data_loader['target']:
                    tgt_data = tgt_data.as_in_context( self.context ) # Target domain only, without labels
                    break
                
                with autograd.record():
                    # supervised on source domain
                    logP_c  = self.model(data)
                    loss        = nd.mean( self.crossEntropy(label_hot, logP_c) )

                    # unsupervised on target domain
                    logP_target = self.model( tgt_data )
                    loss        = loss + nd.mean( self.crossEntropy(nd.exp(logP_target), logP_target) )
                    
                    loss.backward() # back propagation
                self.bnclass_trainer.step(1)

                epoch_loss          += loss.asscalar()
                cumulative_points   += n_points
                
            # log at completion of epoch
            tracelog.logPerformanceData(self, domainTests, e, cumulative_points, epoch_loss/(1+i))
        return tracelog
    
    def trainSequence(self, source_data_loader, target_data_loader, epochs=5):
        adapt_data_loaders, domainTests = self.initForTraining(source_data_loader, target_data_loader)
        
        # co-train source domain supervised, target domain unsupervised
        trace_log = self.trainOptimization(adapt_data_loaders, epochs, domainTests)

        return trace_log


