##
# Domain Adaptation - Batch Normalization Classifier

import mxnet as mx
import numpy as np
import random
import xlsxwriter
from adaptation_methods.adapt_BNC   import BNC, BNC_Cotrained
from adaptation_methods.base_model  import TrainerTraceLog
from dataset_loading.load_datasets import OfficeHomeDatasets


idx_experiment = 0 # init worksheet column for log

def runOffceHomeTasks():
    # OfficeHome Adaptation

    # Prepare results worksheet
    file_name = 'Adaptation_accuracy_results_OfficeHome.xlsx'
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet('Sheet 1')
    global idx_experiment
    idx_experiment = 0
    worksheet.write_column(0, idx_experiment, ['Method', 'Domain Shift', 'Adapted', 'Adapted Std', 'Source Only', 'Source Only Std'])
    idx_experiment += 1

    # set random seeds
    setState()

    # Get dataset
    ds_officehome  = OfficeHomeDatasets()

    # Measure accuracy for each domain shift

    # Run all domain shift pairs
    for src in ds_officehome.domains.keys():
        target = [x for x in ds_officehome.domains.keys() if x is not src]
        for tgt in target:
            accuracy_log, adapt_experiment = testTrainerStochatically(src, tgt, BNC, ds_officehome, worksheet)

    workbook.close()

def setState():
    random.seed(700)
    np.random.seed(seed=701)
    mx.random.seed(702, ctx=mx.cpu(0))

def testTrainerStochatically(source, target, MethodAdapt, multidomain_dataset, worksheet, trials=3):
    tracelog = TrainerTraceLog()
    print(MethodAdapt.__name__ + ' ' + source + '_' + target)
    context = mx.cpu()

    for i in range(trials):
        src = multidomain_dataset.domains[source]
        tgt = multidomain_dataset.domains[target]
        k = multidomain_dataset.k_classes

        experiment = MethodAdapt(k, context)
        data_log = experiment.trainSequence(src, tgt)

        tracelog.stack(data_log)

    # get summary from raw results
    summary_metrics = tracelog.saveRawDataToXLSX(MethodAdapt.__name__ + '_' + source + '_' + target + '.xlsx')
    # write performance summary to excel file
    global idx_experiment
    worksheet.write_column(0, idx_experiment,
                           [MethodAdapt.__name__, source + '-' + target, summary_metrics['adapted_acc'],
                            summary_metrics['adapted_std'],
                            summary_metrics['source_only'], summary_metrics['source_std']])
    idx_experiment += 1
    return tracelog, experiment


def main():
    runOffceHomeTasks()

if __name__ == "__main__":
    main()
