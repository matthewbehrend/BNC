import numpy as np
import mxnet as mx
from mxnet import nd, gluon
from abc import ABC, abstractmethod
import xlsxwriter


class BaseEncoderClass(ABC):
    def __init__(self, k_classes, context=mx.cpu()):
        super().__init__()
        self.k_classes = k_classes
        
        self.context = context
        self.base_learning_rate = 1e-3

    def classifierHead(self):
        net = gluon.nn.Sequential()
        lk = 0.2
        with net.name_scope():
            net.add(gluon.nn.Flatten())

            # fully connected layer 1
            net.add(gluon.nn.Dense( 512 ))
            net.add(gluon.nn.LeakyReLU(lk))
            net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=True))
            net.add(gluon.nn.Dropout(.5))

            # fully conected layer 2
            net.add(gluon.nn.Dense( self.k_classes ))
        return net
    
    def addBatchNorm(self, net):
        # for normalizing the classification layer
        net.add(gluon.nn.BatchNorm(axis=1, center=True, scale=False))
        return net

    def addSoftmax(self, net):
        net.add(gluon.nn.Lambda('log_softmax'))
        return net

    def crossEntropy(self, p, log_q):
        return -nd.sum(p * log_q, axis=1, keepdims=True)   

    def lossBetaSequence(self, p, log_q):
        tol = 1e-5
        return -nd.sum(p * log_q + (1-p) * nd.log(tol + 1 - nd.exp(log_q)), axis=1, keepdims=True)
            
    def printDimensions(net, data_sample):
        # print data dimensions at each layer
        x = data_sample
        layer_thick = 3
        for i in range(len(net) // layer_thick):
            print('Layer ', i+1, ' input shape ', x.shape)
            x = net[layer_thick * i](x)
        print('Output shape ', net(data_sample).shape)
    
    def initForTraining(self, source_data_loader, target_data_loader):
        adapt_data_loaders = {'source':source_data_loader.train, 'target':target_data_loader.train}
        
        domainTests = DomainAccuracyTester(source_data_loader, target_data_loader)    
    
        for i, (data, label) in enumerate(source_data_loader.test):
            self.initTrainer( data )
            break
        return adapt_data_loaders, domainTests
        
    @abstractmethod
    def initTrainer(self, data_sample):
        pass
    @abstractmethod
    def trainOptimization(self, data_loader, n_epochs, domainTests):    
        pass
    @abstractmethod
    def trainSequence(self, source_data_loader, target_data_loader, epochs):
        pass
    @abstractmethod
    def model(self, data):
        pass


class TrainerTraceLog(object):
    def __init__(self):
        self.epoch              = []
        self.n_data_points      = []
        self.source_accuracy    = [] # source-domain accuracy is valid for digits dataset that have train/test splits. OfficeHome is only split by domain, so the source-domain accuracy is just the training accuracy
        self.target_accuracy    = []
        self.avg_batch_loss     = []
        self.train_accuracy     = []
        
    def logPerformanceData(self, adapt_method_obj, accuracyTester, epoch, n_data_points, avg_batch_loss, measure_overfit=True):
        acc_source, acc_target, acc_train = accuracyTester.testAllDomains(adapt_method_obj, measure_overfit)
        self.source_accuracy.append(acc_source)
        self.target_accuracy.append(acc_target)
        self.train_accuracy.append(acc_train)
        self.epoch.append(epoch)
        self.n_data_points.append(n_data_points)
        self.avg_batch_loss.append(avg_batch_loss)
        self.printLatest()
        
    def printLatest(self):
        print("SrcAcc %0.2f, TarAcc %0.2f, Epoch %s Npts %s. TrainAcc %0.2f, loss %0.2f" % 
              (self.source_accuracy[-1], self.target_accuracy[-1], 
               self.epoch[-1], self.n_data_points[-1], 
               self.train_accuracy[-1], self.avg_batch_loss[-1]))

    def append(self, x):
        self.epoch              = self.epoch           + x.epoch
        self.n_data_points      = self.n_data_points   + x.n_data_points
        self.source_accuracy    = self.source_accuracy + x.source_accuracy
        self.target_accuracy    = self.target_accuracy + x.target_accuracy
        self.train_accuracy     = self.train_accuracy  + x.train_accuracy
        self.avg_batch_loss     = self.avg_batch_loss  + x.avg_batch_loss

    def stack(self, x):
        self.epoch              = self.epoch           + [x.epoch]
        self.n_data_points      = self.n_data_points   + [x.n_data_points]
        self.source_accuracy    = self.source_accuracy + [x.source_accuracy]
        self.target_accuracy    = self.target_accuracy + [x.target_accuracy]
        self.train_accuracy     = self.train_accuracy  + [x.train_accuracy]
        self.avg_batch_loss     = self.avg_batch_loss  + [x.avg_batch_loss] 

    def saveRawDataToXLSX(self, file_name):
        workbook = xlsxwriter.Workbook(file_name)
        
        # Create worksheets
        attr_list = ['epoch', 'n_data_points', 'source_accuracy', 'target_accuracy', 'train_accuracy', 'avg_batch_loss']
        for i in range(len(attr_list)):
            worksheet = workbook.add_worksheet(attr_list[i])

            trace = np.array( getattr(self, attr_list[i]) )
            for col_num in range(trace.shape[1]):
                worksheet.write_column(0, col_num, trace[:, col_num].tolist())

        # Model performance at the completion of source-domain training
        idx = self.findLastMin(self.epoch[0])
        source_only_tgt = np.asarray(self.target_accuracy)[:, idx]
        source_only_src = np.asarray(self.source_accuracy)[:, idx]
        
        # Model performance after the adaptation phase is complete
        # Record acuracy in both domains to also assess forgetting of the source domain
        worksheet = workbook.add_worksheet('final_accuracy')
        tgt = np.array(self.target_accuracy)[:, -1] # final adaptation epoch
        src = np.array(self.source_accuracy)[:, -1] # final adaptation epoch

        worksheet.write_column(0, 0, ['adapted_accuracy_tgt_mu', 'adapted_accuracy_tgt_std',  'adapted_accuracy_src_mu', 'adapted_accuracy_src_std',
                                      'src_only_accuracy_tgt_mu', 'src_only_accuracy_tgt_std', 'src_only_accuracy_src_mu', 'src_only_accuracy_src_std'])
        worksheet.write_column(0, 1, [np.mean(tgt),               np.std(tgt, ddof=1),         np.mean(src),              np.std(src, ddof=1),
                                      np.mean(source_only_tgt),   np.std(source_only_tgt, ddof=1), np.mean(source_only_src),   np.std(source_only_src, ddof=1)])
        workbook.close()

        summary_metrics = {'adapted_acc':np.mean(tgt), 'adapted_std':np.std(tgt, ddof=1), 
                           'source_only':np.mean(source_only_tgt), 'source_std':np.std(source_only_tgt, ddof=1),
                           'limit':np.mean(source_only_src), 'limit_std':np.std(source_only_src, ddof=1)}
        return summary_metrics

    def findLastMin(self, x):
        rev_x = np.asarray(x[::-1])
        i = len(rev_x) - np.argmin(rev_x) - 1
        return i
        
    
class DomainAccuracyTester(object):
    def __init__(self, source_dataset, target_dataset):

        self.source_train = source_dataset.train # for measuring overfitting of the data
        self.source = source_dataset.test
        self.target = target_dataset.test
        
    def testAllDomains(self, adapt_method_obj, measure_overfit):
        avg_accuracy_source, avg_c_max_source = DomainAccuracyTester.evaluate_accuracy(self.source, adapt_method_obj.model, adapt_method_obj.k_classes)
        avg_accuracy_target, avg_c_max_target = DomainAccuracyTester.evaluate_accuracy(self.target, adapt_method_obj.model, adapt_method_obj.k_classes)

        avg_accuracy_training = 0
        if(measure_overfit):
            avg_accuracy_training, avg_c_max_training = DomainAccuracyTester.evaluate_accuracy(self.source_train, adapt_method_obj.model, adapt_method_obj.k_classes)
        
        return avg_accuracy_source, avg_accuracy_target, avg_accuracy_training

    def evaluate_accuracy(data_loader, model, kclasses):
    
        if(data_loader == None):
            return (0, 0)
        
        n_accum = 0.0
        accuracy_accum = 0.0
        
        for i, (data, label) in enumerate(data_loader):
            data  =  data.as_in_context( mx.cpu() )
            label = label.as_in_context( mx.cpu() )

            log_c = model(data)
            
            # standard practice classification accuracy
            accuracy = (nd.argmax(log_c, axis=1).astype('int32') == label.astype('int32')).astype('float32')
    
            accuracy_accum += nd.sum(accuracy)
            n_accum += len(label)
            
        avg_accuracy    = (accuracy_accum / n_accum).asscalar()
        avg_c_max       = nd.exp(nd.max(log_c, axis=1)).mean().asscalar() # maximum value in classification vector      
        return (avg_accuracy, avg_c_max)
       



