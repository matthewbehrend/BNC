##
# Domain Adaptation - Batch Normalization Classifier
# inspecting the distribution at the input to softmax

# some tech refs
# https://stackoverflow.com/questions/36815115/matrix-elements-for-scatter-plot 

import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from adaptation_methods.adapt_BNC_inspect_distributions   import BNCInspect
from dataset_loading.load_datasets import OfficeHomeDatasets

#tsne = TSNE(n_components=2, random_state=0)

context = mx.cpu()

### Helper functions
def setState():    
    random.seed(700)
    np.random.seed(seed=701)
    mx.random.seed(702, ctx=mx.cpu(0))


# set random seeds
setState()

#how mane epochs to train
epk = 5

# get dataset
ds_feat  = OfficeHomeDatasets()

source_name = 'product'
target_name = 'art'

ds_full_batch = OfficeHomeDatasets(asdataloader=False)
domain_full_batch = getattr(ds_full_batch, target_name).train
full_batch_data  = nd.array(domain_full_batch._data[0])
full_batch_labels = domain_full_batch._data[1]

# adapt to a specific domain shift
src = ds_feat.domains[source_name]
tgt = ds_feat.domains[target_name]
experiment = BNCInspect( ds_feat.k_classes, context, bypassbn=False)
accuracy_log            = experiment.trainSequence(src, tgt,epochs=epk)


# get the input to softmax, and class label
softmax_inputs  = experiment.get_softmax_input_all_data( full_batch_data ) # (N points x k classes), (N points)
fc_outputs = experiment.get_fc_output_all_data( full_batch_data )
labels = full_batch_labels

# retrain without batchnorm
experiment_noBN = BNCInspect( ds_feat.k_classes, context, bypassbn=True)
accuracy_log_noBN                   = experiment_noBN.trainSequence(src, tgt,epochs=epk)
softmax_inputs_noBN    = experiment_noBN.get_softmax_input_all_data( full_batch_data ) # (N points x k classes), (N points)
fc_outputs_noBN    = experiment_noBN.get_fc_output_all_data( full_batch_data )
labels_noBN = full_batch_labels

bins = np.linspace(-0.1, 0.175, 50)
        
d = experiment.get_fc_tensor_all_data( full_batch_data )
d_noBN = experiment_noBN.get_fc_tensor_all_data( full_batch_data )

plt.hist(d, bins, alpha=0.5, label='Tensor weights with BN',density=False)
plt.hist(d_noBN, bins, alpha=0.5, label='Tensor weights without BN',density=False)
#plt.yscale('log', nonposy='clip')
#plt.ylim([0.1,40.0])
plt.xlabel('Output Values')
plt.ylabel('Counts, arbitrary units')
plt.legend(loc='upper right')
plt.title('Trained tensor weights, Dense layer 1, ' + str(int(epk)) + ' epochs')
plt.savefig('./plots/TensorWeights_' + str(int(epk))  + '_epochs.jpg',format="jpg")
#plt.show()
plt.close()

# A few quantites to use for the visualizations
numclasses = len(softmax_inputs_noBN[0,:])
colors = cm.rainbow(np.linspace(0, 1, numclasses))

##### HISTOGRAM PLOTS ########
#bins = np.linspace(-10, 10, 50)
bins = np.linspace(-10, 10, 100)

# for the channel-by-channel histograms (choose objects)
# [bed, chair, couch, alarm clock, fan, bottle]
# class_list = [3,10,13,0,19,5]
class_list = [0,1,2,3]
fig, axs = plt.subplots(len(class_list), sharex=True, sharey=True, gridspec_kw={'hspace': 0})
#for classnum in range(6):
for classnum_iter in range(len(class_list)):
    classnum = class_list[classnum_iter]
    thisclass_si = softmax_inputs[:,classnum]
    thisclass_si_correct = thisclass_si[np.where(labels==classnum)]
    thisclass_si_incorrect = thisclass_si[np.where(labels!=classnum)]
    thisclass_si_noBN = softmax_inputs_noBN[:,classnum]
    thisclass_si_correct_noBN = thisclass_si_noBN[np.where(labels_noBN==classnum)]
    thisclass_si_incorrect_noBN = thisclass_si_noBN[np.where(labels_noBN!=classnum)]

    thisclass_fo = fc_outputs[:,classnum]
    thisclass_fo_correct = thisclass_fo[np.where(labels==classnum)]
    thisclass_fo_incorrect = thisclass_fo[np.where(labels!=classnum)]
    thisclass_fo_noBN = fc_outputs_noBN[:,classnum]
    thisclass_fo_correct_noBN = thisclass_fo_noBN[np.where(labels_noBN==classnum)]
    thisclass_fo_incorrect_noBN = thisclass_fo_noBN[np.where(labels_noBN!=classnum)]

    #This makes plots one by one.  It's fine but a little cumbersome to copy them all.
    #just turn this on if you want them:
    if True:
        # plt.hist(thisclass_si_correct, bins, alpha=0.5, label='Correct Class, with BN',density=True)
        # plt.hist(thisclass_si_correct_noBN, bins, alpha=0.5, label='Correct Class, without BN',density=True)
        # plt.hist(thisclass_si_incorrect, bins, alpha=0.5, label='Incorrect Class, with BN',density=True)
        # plt.hist(thisclass_si_incorrect_noBN, bins, alpha=0.5, label='Incorrect Class, without BN',density=True)
        # plt.xlabel('Weight')
        # plt.ylabel('Counts, arbitrary units')
        # plt.legend(loc='upper right')
        # plt.title('SM Inputs, channel ' + str(classnum))
        # plt.show()
        if classnum == 0:
            axs[classnum_iter].hist(thisclass_fo_correct, bins, alpha=0.5, label='Correct Class, with BN',density=True)
            axs[classnum_iter].hist(thisclass_fo_correct_noBN, bins, alpha=0.5, label='Correct Class, without BN',density=True)
            axs[classnum_iter].hist(thisclass_fo_incorrect, bins, alpha=0.5, label='Incorrect Class, with BN',density=True)
            axs[classnum_iter].hist(thisclass_fo_incorrect_noBN, bins, alpha=0.5, label='Incorrect Class, without BN',density=True)
        else:
            axs[classnum_iter].hist(thisclass_fo_correct, bins, alpha=0.5, label='_nolegend_',density=True)
            axs[classnum_iter].hist(thisclass_fo_correct_noBN, bins, alpha=0.5, label='_nolegend_',density=True)
            axs[classnum_iter].hist(thisclass_fo_incorrect, bins, alpha=0.5, label='_nolegend_',density=True)
            axs[classnum_iter].hist(thisclass_fo_incorrect_noBN, bins, alpha=0.5, label='_nolegend_',density=True)
            #plt.yscale('log', nonpositive='clip')

        #plt.hist(thisclass_fo_correct, bins, alpha=0.5, label='Correct Class, with BN',density=True)
        #plt.hist(thisclass_fo_correct_noBN, bins, alpha=0.5, label='Correct Class, without BN',density=True)
        #plt.hist(thisclass_fo_incorrect, bins, alpha=0.5, label='Incorrect Class, with BN',density=True)
        #plt.hist(thisclass_fo_incorrect_noBN, bins, alpha=0.5, label='Incorrect Class, without BN',density=True)
        #plt.xlabel('Weight')
        #plt.ylabel('Counts, arbitrary units')
        #plt.legend(loc='upper left')
        # plt.title('Channel ' + str(classnum))
        #axs[classnum].title('FC Outputs, channel ' + str(classnum))

fig.legend(loc='upper left')
fig.text(0.5, 0.04, 'Channel Value', ha='center')
fig.text(0.04, 0.5, 'Counts, arbitrary units', va='center', rotation='vertical')
for ax in axs:
    ax.label_outer()
#plt.show()
plt.savefig('./plots/Hist_FCOutputs_bychannel.jpg',format="jpg")
plt.close()

if True:
    #combining all the outputs into one set of histograms
    plt.hist(np.concatenate([softmax_inputs[np.where(labels==c),c][0] for c in range(numclasses)]), bins, alpha=0.5, label='Correct Class, with BN',density=True)
    plt.hist(np.concatenate([softmax_inputs_noBN[np.where(labels_noBN==c),c][0] for c in range(numclasses)]), bins, alpha=0.5, label='Correct Class, without BN',density=True)
    plt.hist(np.concatenate([softmax_inputs[np.where(labels==c),:][0][0] for c in range(numclasses)]), bins, alpha=0.5, label='Incorrect Class, with BN',density=True)
    plt.hist(np.concatenate([softmax_inputs_noBN[np.where(labels_noBN==c),:][0][0] for c in range(numclasses)]), bins, alpha=0.5, label='Incorrect Class, without BN',density=True)
    plt.xlabel('Weight')
    plt.ylabel('Counts, arbitrary units')
    plt.legend(loc='upper right')
    plt.title('All SM Inputs')
    plt.savefig('./plots/Hist_SMInputs.jpg',format="jpg")
    #plt.show()
    plt.close()

    plt.hist(np.concatenate([fc_outputs[np.where(labels==c),c][0] for c in range(numclasses)]), bins, alpha=0.5, label='Correct Class, with BN',density=True)
    plt.hist(np.concatenate([fc_outputs_noBN[np.where(labels_noBN==c),c][0] for c in range(numclasses)]), bins, alpha=0.5, label='Correct Class, without BN',density=True)
    plt.hist(np.concatenate([fc_outputs[np.where(labels==c),:][0][0] for c in range(numclasses)]), bins, alpha=0.5, label='Incorrect Class, with BN',density=True)
    plt.hist(np.concatenate([fc_outputs_noBN[np.where(labels_noBN==c),:][0][0] for c in range(numclasses)]), bins, alpha=0.5, label='Incorrect Class, without BN',density=True)
    plt.xlabel('Weight')
    plt.ylabel('Counts, arbitrary units')
    plt.legend(loc='upper right')
    plt.title('All FC Outputs')
    plt.savefig('./plots/Hist_FCOutputs.jpg',format="jpg")
    #plt.show()
    plt.close()

#####SOME REAL SIMPLE PLOTS - ALL THE DATA BY CHANNEL:
if False:
    figs, axes = plt.subplots(2)

    # repeat the channel vector for the amount of images we have
    x = np.repeat(range(softmax_inputs.shape[1]),softmax_inputs.shape[0])
    # now reshape the a matrix to generate a vector
    y = np.reshape(softmax_inputs.T,(1,np.product(softmax_inputs.shape) ))

    ax = axes[0]
    ax.scatter(x, y, s=2)
    ax.set_title('Raw SM inputs, with BatchNorm')
    ax.set_xlabel('label')
    ax.set_ylabel('score')

    x = np.repeat(range(softmax_inputs_noBN.shape[1]),softmax_inputs_noBN.shape[0])
    y = np.reshape(softmax_inputs_noBN.T,(1,np.product(softmax_inputs_noBN.shape) ))

    ax = axes[1]
    ax.scatter(x, y, s=2)
    ax.set_title('Raw SM inputs, without BatchNorm')
    ax.set_xlabel('label')
    ax.set_ylabel('score')

    plt.draw
    plt.show()


    figs, axes = plt.subplots(2)

    # repeat the channel vector for the amount of images we have
    x = np.repeat(range(fc_outputs.shape[1]),fc_outputs.shape[0])
    # now reshape the a matrix to generate a vector
    y = np.reshape(fc_outputs.T,(1,np.product(fc_outputs.shape) ))

    ax = axes[0]
    ax.scatter(x, y, s=2)
    ax.set_title('Raw FC outputs, with BatchNorm')
    ax.set_xlabel('label')
    ax.set_ylabel('score')

    x = np.repeat(range(fc_outputs_noBN.shape[1]),fc_outputs_noBN.shape[0])
    y = np.reshape(fc_outputs_noBN.T,(1,np.product(fc_outputs_noBN.shape) ))

    ax = axes[1]
    ax.scatter(x, y, s=2)
    ax.set_title('Raw FC outputs, without BatchNorm')
    ax.set_xlabel('label')
    ax.set_ylabel('score')

    plt.draw
    plt.show()




##### T-SNE PLOTS ########

if True:

    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(softmax_inputs_noBN)
    plt.figure(figsize=(6, 6))
    for l, c, co, in zip([str(i) for i in range(numclasses)], colors, range(numclasses)):
        plt.scatter(tsne_data[np.where(labels_noBN == co), 0],
                    tsne_data[np.where(labels_noBN == co), 1],
                    marker='o',
                    color=c,
                    linewidth=0.6,
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on trained-net SM inputs, No BN')
    plt.savefig('./plots/TSNE_SMI_NoBN.jpg',format="jpg")
    #plt.legend(loc='best')
    #plt.show()
    plt.close()

    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(softmax_inputs)
    plt.figure(figsize=(6, 6))
    for l, c, co, in zip([str(i) for i in range(numclasses)], colors, range(numclasses)):
        plt.scatter(tsne_data[np.where(labels_noBN == co), 0],
                    tsne_data[np.where(labels_noBN == co), 1],
                    marker='o',
                    color=c,
                    linewidth=0.6,
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on trained-net SM inputs, With BN')
    plt.savefig('./plots/TSNE_SMI_WithBN.jpg',format="jpg")
    #plt.legend(loc='best')
    #plt.show()
    plt.close()


    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(fc_outputs_noBN)
    plt.figure(figsize=(6, 6))
    for l, c, co, in zip([str(i) for i in range(numclasses)], colors, range(numclasses)):
        plt.scatter(tsne_data[np.where(labels_noBN == co), 0],
                    tsne_data[np.where(labels_noBN == co), 1],
                    marker='o',
                    color=c,
                    linewidth=0.6,
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on trained-net FC outputs, No BN')
    plt.savefig('./plots/TSNE_FCO_NoBN.jpg',format="jpg")
    #plt.legend(loc='best')
    #plt.savefig('rainbow-01.png')
    #plt.show()
    plt.close()

    tsne = TSNE(n_components=2, random_state=0)
    tsne_data = tsne.fit_transform(fc_outputs)
    plt.figure(figsize=(6, 6))
    for l, c, co, in zip([str(i) for i in range(numclasses)], colors, range(numclasses)):
        plt.scatter(tsne_data[np.where(labels_noBN == co), 0],
                    tsne_data[np.where(labels_noBN == co), 1],
                    marker='o',
                    color=c,
                    linewidth=0.6,
                    alpha=0.8,
                    label=l)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on trained-net FC outputs, With BN')
    plt.savefig('./plots/TSNE_FCO_WithBN.jpg',format="jpg")
    #plt.legend(loc='best')
    #plt.show()
    plt.close()

##### INTRA-CHANNEL MEANS#########

if False:
    # SM Input
    figs, axes = plt.subplots(2, 2)

    ax = axes[0, 0]
    y = np.mean(softmax_inputs, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_title('SM input, intrachannel means, with BN')
    ax.set_ylabel('mean')

    ax = axes[1, 0]
    y = np.std(softmax_inputs, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_ylabel('std')
    ax.set_xlabel('label')

    ax = axes[0, 1]
    y = np.mean(softmax_inputs_noBN, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_title('SM input, intrachannel means, no BN')
    ax.set_ylabel('mean')

    ax = axes[1, 1]
    y = np.std(softmax_inputs_noBN, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_ylabel('std')
    ax.set_xlabel('label')

    plt.draw
    plt.show()


    # FC Output
    figs, axes = plt.subplots(2, 2)

    ax = axes[0, 0]
    y = np.mean(fc_outputs, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_title('FC output, intrachannel means, with BN')
    ax.set_ylabel('mean')

    ax = axes[1, 0]
    y = np.std(fc_outputs, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_ylabel('std')
    ax.set_xlabel('label')

    ax = axes[0, 1]
    y = np.mean(fc_outputs_noBN, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_title('FC output, intrachannel means, no BN')
    ax.set_ylabel('mean')

    ax = axes[1, 1]
    y = np.std(fc_outputs_noBN, axis=0)
    ax.scatter(range(numclasses), y, s=2)
    ax.set_ylabel('std')
    ax.set_xlabel('label')

    plt.draw
    plt.show()



######### SCORE PLOTS ###############
if False:
    # Originals
    figs, axes = plt.subplots(2, 2)
    print(softmax_inputs)
    print(np.mean(softmax_inputs, axis=1))
    print(len(np.mean(softmax_inputs, axis=1)))
    print([i for i in labels])

    ax = axes[0, 0]
    y = np.mean(softmax_inputs, axis=1)
    ax.scatter(labels, y, s=2)
    ax.set_title('BatchNorm')
    ax.set_ylabel('mean')

    ax = axes[1, 0]
    y = np.std(softmax_inputs, axis=1)
    ax.scatter(labels, y, s=2)
    ax.set_ylabel('std')
    ax.set_xlabel('label')


    y = np.mean(softmax_inputs_noBN, axis=1)
    ax = axes[0, 1]
    ax.scatter(labels_noBN, y, s=2)
    ax.set_title('Without BatchNorm')

    y = np.std(softmax_inputs_noBN, axis=1)
    ax = axes[1, 1]
    ax.scatter(labels_noBN, y, s=2)
    ax.set_xlabel('label')

    plt.draw
    plt.show()

    # FC-outputs:
    figs, axes = plt.subplots(2, 2)

    ax = axes[0, 0]
    y = np.mean(fc_outputs, axis=1)
    ax.scatter(labels, y, s=2)
    ax.set_title('FC outputs, with BatchNorm')
    ax.set_ylabel('mean')

    ax = axes[1, 0]
    y = np.std(fc_outputs, axis=1)
    ax.scatter(labels, y, s=2)
    ax.set_ylabel('std')
    ax.set_xlabel('label')


    y = np.mean(fc_outputs_noBN, axis=1)
    ax = axes[0, 1]
    ax.scatter(labels_noBN, y, s=2)
    ax.set_title('FC outputs, Without BatchNorm')

    y = np.std(fc_outputs_noBN, axis=1)
    ax = axes[1, 1]
    ax.scatter(labels_noBN, y, s=2)
    ax.set_xlabel('label')

    plt.draw
    plt.show()

if False:
    #also take a look at the scores for winning classes
    figs, axes = plt.subplots(1, 2)
    ax = axes[0]
    y = [softmax_inputs[i,labels[i]] for i in range(len(softmax_inputs[:,0]))]
    ax.scatter(labels, y, s=2)
    ax.set_title('BatchNorm')
    ax.set_ylabel('correct label scores')

    ax = axes[1]
    y = [softmax_inputs_noBN[i,labels_noBN[i]] for i in range(len(softmax_inputs_noBN[:,0]))]
    ax.scatter(labels, y, s=2)
    ax.set_title('No BatchNorm')
    ax.set_ylabel('correct label scores')

    plt.draw
    plt.show()


    #and we'll look at how much above the mean the correct class is:
    figs, axes = plt.subplots(1, 2)
    ax = axes[0]
    y = [softmax_inputs[i,labels[i]] for i in range(len(softmax_inputs[:,0]))] - np.mean(softmax_inputs, axis=1)
    ax.scatter(labels, y, s=2)
    ax.set_title('BatchNorm')
    ax.set_ylabel('correct label scores above mean ')

    ax = axes[1]
    y = [softmax_inputs_noBN[i,labels_noBN[i]] for i in range(len(softmax_inputs_noBN[:,0]))] - np.mean(softmax_inputs_noBN, axis=1)
    ax.scatter(labels, y, s=2)
    ax.set_title('No BatchNorm')
    ax.set_ylabel('correct label scores above mean')

    plt.draw
    plt.show()




