import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr


from manual_training_inference import *

def get_str(params):
    if params['AU']:
        return '_AU'
    elif params['EU']:
        return '_EU'
    else:
        return '_X'

def plot_percentage_target_EU(epis_uncertainty, target, params):

    x = np.arange(100)

    target_proportion = np.sum(target)/len(target)
    sorted_epis_uncertainty = np.argsort(epis_uncertainty)
    # The distribution of target = 1 as sorted by epis_uncertainty
    target_distribution_epis_uncertainty = np.zeros(len(target))
    for i in range(len(target)):
        if target[sorted_epis_uncertainty[i]] == 1:
            target_distribution_epis_uncertainty[i] = 1

    percentage_target_distribution_epis_uncertainty = np.zeros(100)
    lht_epis_uncertainty = np.sort(epis_uncertainty)
    low_high_epis_uncertainty = np.zeros(100)

    for percentage in range(100):
        start_index = int(percentage*len(target)/100)
        last_index = int((percentage+1)*len(target)/100)
        percentage_target_distribution_epis_uncertainty[percentage] = \
            np.sum(target_distribution_epis_uncertainty[start_index:last_index]) / len(target_distribution_epis_uncertainty[start_index:last_index])
        low_high_epis_uncertainty[percentage] = np.sum(lht_epis_uncertainty[start_index:last_index]) / len(lht_epis_uncertainty[start_index:last_index])


    plt.plot(x,[target_proportion]*len(x), label='Average Target Proportion', color='green')
    plt.plot(x,percentage_target_distribution_epis_uncertainty, label='Target Proportion sorted by Epistemic Uncertainty', color='red')
    pearson = pearsonr(percentage_target_distribution_epis_uncertainty, low_high_epis_uncertainty)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.title('Comparison of average target proportion and target proportion as sorted by epistemic uncertainty', wrap=True)
    plt.legend()
    plt.savefig('Figures/Epistemic_uncertainty_target_proportion_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return target_proportion,percentage_target_distribution_epis_uncertainty

def plot_percentage_target_AU(aleatoric_uncertainty, target, params):

    x = np.arange(100)

    target_proportion = np.sum(target)/len(target)
    sorted_epis_uncertainty = np.argsort(aleatoric_uncertainty)
    # The distribution of target = 1 as sorted by epis_uncertainty
    target_distribution_epis_uncertainty = np.zeros(len(target))
    for i in range(len(target)):
        if target[sorted_epis_uncertainty[i]] == 1:
            target_distribution_epis_uncertainty[i] = 1

    percentage_target_distribution_epis_uncertainty = np.zeros(100)
    lht_epis_uncertainty = np.sort(aleatoric_uncertainty)
    low_high_epis_uncertainty = np.zeros(100)

    for percentage in range(100):
        start_index = int(percentage*len(target)/100)
        last_index = int((percentage+1)*len(target)/100)
        percentage_target_distribution_epis_uncertainty[percentage] = \
            np.sum(target_distribution_epis_uncertainty[start_index:last_index]) / len(target_distribution_epis_uncertainty[start_index:last_index])
        low_high_epis_uncertainty[percentage] = np.sum(lht_epis_uncertainty[start_index:last_index]) / len(lht_epis_uncertainty[start_index:last_index])


    plt.plot(x,[target_proportion]*len(x), label='Average Target Proportion', color='green')
    plt.plot(x,percentage_target_distribution_epis_uncertainty, label='Target Proportion sorted by Aleatoric Uncertainty', color='red')
    pearson = pearsonr(percentage_target_distribution_epis_uncertainty, low_high_epis_uncertainty)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.title('Comparison of average target proportion and target proportion as sorted by aleatoric uncertainty', wrap=True)
    plt.legend()
    plt.savefig('Figures/Aleatoric_uncertainty_target_proportion_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return target_proportion,percentage_target_distribution_epis_uncertainty

def plot_percentage_target_TU(total_uncertainty, target, params):

    x = np.arange(100)

    target_proportion = np.sum(target)/len(target)
    sorted_epis_uncertainty = np.argsort(total_uncertainty)
    # The distribution of target = 1 as sorted by epis_uncertainty
    target_distribution_epis_uncertainty = np.zeros(len(target))
    for i in range(len(target)):
        if target[sorted_epis_uncertainty[i]] == 1:
            target_distribution_epis_uncertainty[i] = 1

    percentage_target_distribution_epis_uncertainty = np.zeros(100)
    lht_epis_uncertainty = np.sort(total_uncertainty)
    low_high_epis_uncertainty = np.zeros(100)

    for percentage in range(100):
        start_index = int(percentage*len(target)/100)
        last_index = int((percentage+1)*len(target)/100)
        percentage_target_distribution_epis_uncertainty[percentage] = \
            np.sum(target_distribution_epis_uncertainty[start_index:last_index]) / len(target_distribution_epis_uncertainty[start_index:last_index])
        low_high_epis_uncertainty[percentage] = np.sum(lht_epis_uncertainty[start_index:last_index]) / len(lht_epis_uncertainty[start_index:last_index])


    plt.plot(x,[target_proportion]*len(x), label='Average Target Proportion', color='green')
    plt.plot(x,percentage_target_distribution_epis_uncertainty, label='Target Proportion sorted by Total Uncertainty', color='red')
    pearson = pearsonr(percentage_target_distribution_epis_uncertainty, low_high_epis_uncertainty)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.title('Comparison of average target proportion and target proportion as sorted by total uncertainty', wrap=True)
    plt.legend()
    plt.savefig('Figures/Total_uncertainty_target_proportion_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return target_proportion,percentage_target_distribution_epis_uncertainty


def plot_correlation_AU_TU(exp_entropy, pred_entropy, params):

    plt.plot(np.arange(len(exp_entropy)),exp_entropy, label='Aleatoric Uncertainty', color='green')
    plt.plot(np.arange(len(pred_entropy)),pred_entropy, label='Total Uncertainty', color='red')
    pearson = pearsonr(exp_entropy, pred_entropy)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.title('Comparison of expected entropy and predictive entropy', wrap=True)
    plt.legend()
    plt.savefig('Figures/Total_uncertainty_aleatoric_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return pearson



def plot_percentage_AU(aleatoric_acc, aleatoric_true, aleatoric_false, exp_entropy, params):

    x = np.arange(100)
    exp_entropy = np.sort(exp_entropy)
    # average aleatoric
    aleatoric_percentage = np.zeros(100)
    for percentage in range(100):
        start_index = int(percentage*len(exp_entropy)/100)
        last_index = int((percentage+1)*len(exp_entropy)/100)
        aleatoric_percentage[percentage] = \
            np.sum(exp_entropy[start_index:last_index]) / len(exp_entropy[start_index:last_index])


    plt.plot(x,aleatoric_acc, label= 'Accuracy by Aleatoric uncertainty', color='blue')
    #plt.plot(x,aleatoric_true, label= 'Total Aleatoric uncertainty of correct predictions', color='green')
    #plt.plot(x,aleatoric_false, label= 'Total Aleatoric uncertainty of wrong predictions', color='red')
    plt.plot(x,aleatoric_percentage, label= 'Aleatoric uncertainty')
    plt.title('Accuracy and Aleatoric Uncertainty by percentage of data', wrap=True)
    pearson = pearsonr(aleatoric_percentage, aleatoric_acc)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.legend()
    plt.savefig('Figures/Aleatoric_uncertainty_acc_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return 0





def plot_diff_TU(clean_pred_entropy,clean_true_labels, clean_pred_labels, testacc, testf1, testprecision, testrecall, params):

    uncertain_change_f1 = []
    uncertain_change_acc = []
    uncertain_change_precision = []
    uncertain_change_recall = []

    highest_uncertain_change_f1 = []
    highest_uncertain_change_acc = []
    highest_uncertain_change_precision = []
    highest_uncertain_change_recall = []

    # calculate accuracy improvement for removing highest uncertainty outputs
    sorted_pred_entropy = np.argsort(clean_pred_entropy)
    x_boundary = [0.2,0.3,0.45,0.65,0.85,1,1.25,1.5,1.75,2,2.33,2.66,3,3.5,4,4.5,5,5.66,6.5,7.5,8.7, 10,12.5,15,17.5,20,23.33,26.66, 30,35,40,47,55,65,75,87,99]
    for boundary in x_boundary:

        # for every boundary create new uncertain_arrays of given length and calculate metrics
        len_array = int(boundary*len(clean_true_labels)/100)
        uncertain_true_labels = []
        uncertain_pred_labels = []
        for i in sorted_pred_entropy[::-1][:len_array]:
            uncertain_true_labels.append(clean_true_labels[i])
            uncertain_pred_labels.append(clean_pred_labels[i])
        uncertain_change_f1.append(f1_score(uncertain_true_labels, uncertain_pred_labels, average='macro'))
        uncertain_change_acc.append(accuracy_score(uncertain_true_labels,uncertain_pred_labels))
        uncertain_change_precision.append(precision_score(uncertain_true_labels, uncertain_pred_labels, average='macro'))
        uncertain_change_recall.append(recall_score(uncertain_true_labels, uncertain_pred_labels, average='macro'))

        highest_uncertain_true_labels = []
        highest_uncertain_pred_labels = []
        for i in sorted_pred_entropy[:len_array]:
            highest_uncertain_true_labels.append(clean_true_labels[i])
            highest_uncertain_pred_labels.append(clean_pred_labels[i])
        highest_uncertain_change_f1.append(f1_score(highest_uncertain_true_labels, highest_uncertain_pred_labels, average='macro'))
        highest_uncertain_change_acc.append(accuracy_score(highest_uncertain_true_labels,highest_uncertain_pred_labels))
        highest_uncertain_change_precision.append(precision_score(highest_uncertain_true_labels, highest_uncertain_pred_labels, average='macro'))
        highest_uncertain_change_recall.append(recall_score(highest_uncertain_true_labels, highest_uncertain_pred_labels, average='macro'))


        print('Accuracy for the ' + str(boundary) + '% elements with the highest uncertainty: {}'.format(uncertain_change_acc[::-1][0]))
        print('F1 score for the ' + str(boundary) + '% elements with the highest uncertainty: {}'.format(uncertain_change_f1[::-1][0]))
        print('Precision for the ' + str(boundary) + '% elements with the highest uncertainty: {}'.format(uncertain_change_precision[::-1][0]))
        print('Recall for the ' + str(boundary) + '% elements with the highest uncertainty: {}'.format(uncertain_change_recall[::-1][0]))



    plt.plot(x_boundary,uncertain_change_acc, label= 'Subset Accuracy', color='blue')
    plt.plot(x_boundary,uncertain_change_f1, label= 'Subset F1 Score', color='green')
    #plt.plot(x_boundary,[testacc]*len(x_boundary), label= 'Total Accuracy', color='red')
    #plt.plot(x_boundary,[testf1]*len(x_boundary), label= 'Total F1 Score', color='black')
    plt.title('F1 and Accuracy of X% of lowest Total Uncertainty samples')
    plt.xscale('log')
    plt.legend()
    plt.savefig('Figures/Acc_f1_lowest_Total_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    plt.plot(x_boundary,uncertain_change_precision, label= 'Subset Precision', color='blue')
    plt.plot(x_boundary,uncertain_change_recall, label= 'Subset Recall', color='green')
    #plt.plot(x_boundary,[testprecision]*len(x_boundary), label= 'Total Precision', color='red')
    #plt.plot(x_boundary,[testrecall]*len(x_boundary), label= 'Total Recall', color='black')
    plt.title('Precision and Recall of X% of lowest Total Uncertainty samples')
    plt.xscale('log')
    plt.legend()
    plt.savefig('Figures/Precision_recall_lowest_Total_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    plt.plot(x_boundary,highest_uncertain_change_acc, label= 'Subset Accuracy', color='blue')
    plt.plot(x_boundary,highest_uncertain_change_f1, label= 'Subset F1 Score', color='green')
    #plt.plot(x_boundary,[testacc]*len(x_boundary), label= 'Total Accuracy', color='red')
    #plt.plot(x_boundary,[testf1]*len(x_boundary), label= 'Total F1 Score', color='black')
    plt.title('F1 and Accuracy of X% of highest Total Uncertainty samples')
    plt.xscale('log')
    plt.legend()
    plt.savefig('Figures/Acc_f1_highest_Total_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    plt.plot(x_boundary,highest_uncertain_change_precision, label= 'Subset Precision', color='blue')
    plt.plot(x_boundary,highest_uncertain_change_recall, label= 'Subset Recall', color='green')
    #plt.plot(x_boundary,[testprecision]*len(x_boundary), label= 'Total Precision', color='red')
    #plt.plot(x_boundary,[testrecall]*len(x_boundary), label= 'Total Recall', color='black')
    plt.title('Precision and Recall of X% of highest Total Uncertainty samples')
    plt.xscale('log')
    plt.legend()
    plt.savefig('Figures/Precision_recall_highest_Total_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()


    return uncertain_change_acc, uncertain_change_f1, uncertain_change_precision, uncertain_change_recall

# def plot_percentage_TU(total_acc, total_true, total_false,params):
#
#     x = np.arange(100)
#
#     plt.plot(x,total_acc, label= 'Accuracy sorted by Total Uncertainty', color='blue')
#     plt.plot(x,total_true, label= 'Total Uncertainty of correct predictions', color='green')
#     plt.plot(x,total_false, label= 'Total Uncertainty of wrong predictions', color='red')
#     plt.title('Accuracy, Total uncertainty')
#     plt.legend()
#     plt.savefig('Figures/Total_uncertainty_tf_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
#     plt.close()
#
#     return 0

def plot_percentage_TU_total(total_acc, pred_entropy,params):

    x = np.arange(100)
    total_uncertain = np.sort(pred_entropy)
    total_uncertain_percentage = np.zeros(100)

    for percentage in range(100):
        start_index = int(percentage*len(total_uncertain)/100)
        last_index = int((percentage+1)*len(total_uncertain)/100)
        total_uncertain_percentage[percentage] = \
            np.sum(total_uncertain[start_index:last_index]) / len(total_uncertain[start_index:last_index])


    plt.plot(x,total_acc, label= 'Accuracy sorted by Total Uncertainty', color='blue')
    plt.plot(x,total_uncertain_percentage, label= 'Total Uncertainty', color='green')
    plt.title('Accuracy, Total uncertainty')
    pearson = pearsonr(total_uncertain_percentage, total_acc)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.legend()
    plt.savefig('Figures/Total_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return 0


def plot_percentage_EU_tf(epistemic_acc, epistemic_true, epistemic_false, params):

    x = np.arange(100)

    plt.plot(x,epistemic_acc, label= 'Accuracy sorted by Epistemic Uncertainty', color='blue')
    plt.plot(x,epistemic_true, label= 'Epistemic Uncertainty of correct predictions', color='green')
    plt.plot(x,epistemic_false, label= 'Epistemic Uncertainty of wrong predictions', color='red')
    plt.title('Accuracy, Epistemic uncertainty')
    plt.legend()
    plt.savefig('Figures/Epistemic_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return 0

def plot_percentage_EU(epistemic_acc, epis_uncertainty, params):

    x = np.arange(100)
    epis_uncertain = np.sort(epis_uncertainty)
    epis_uncertain_percentage = np.zeros(100)

    for percentage in range(100):
        start_index = int(percentage*len(epis_uncertain)/100)
        last_index = int((percentage+1)*len(epis_uncertain)/100)
        epis_uncertain_percentage[percentage] = \
            np.sum(epis_uncertain[start_index:last_index]) / len(epis_uncertain[start_index:last_index])

    plt.plot(x,epistemic_acc, label= 'Accuracy sorted by Epistemic Uncertainty', color='blue')
    plt.plot(x,epis_uncertain_percentage, label= 'Epistemic Uncertainty', color='red')
    plt.title('Accuracy, Epistemic uncertainty')
    pearson = pearsonr(epis_uncertain_percentage, epistemic_acc)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.legend()
    plt.savefig('Figures/Epistemic_uncertainty_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return 0

def plot_dissent_AU(exp_entropy, dissent, matcharray, params):

    dissent_asc_indeces = np.argsort(dissent)
    new_dissent_level = []

    # Where does 0 go to 1 and 1 to 2?
    for i in range(len(dissent)-1):
        if dissent[dissent_asc_indeces[i+1]] > dissent[dissent_asc_indeces[i]]:
            new_dissent_level.append(i)

    # 0: true 0 dissent, 1: false 0 dissent, 2: true 1 dissent, 3: false 1 dissent
    sum_true_false_dissent = np.zeros(4)

    for i in range(new_dissent_level[1]):
        if i <= new_dissent_level[0]:
            if matcharray[dissent_asc_indeces[i]]:
                sum_true_false_dissent[0]+=1
            else:
                sum_true_false_dissent[1]+=1
        else:
            if matcharray[dissent_asc_indeces[i]]:
                sum_true_false_dissent[2]+=1
            else:
                sum_true_false_dissent[3]+=1

    avg_AU_per_dissent = [0,0,0]

    for i in range(len(dissent)):
        if i <= new_dissent_level[0]:
            avg_AU_per_dissent[0]+=exp_entropy[dissent_asc_indeces[i]]
        elif i <= new_dissent_level[1]:
            avg_AU_per_dissent[1]+=exp_entropy[dissent_asc_indeces[i]]
        else:
            avg_AU_per_dissent[2]+=exp_entropy[dissent_asc_indeces[i]]

    avg_AU_per_dissent[0]=avg_AU_per_dissent[0]/(new_dissent_level[0]+1)
    avg_AU_per_dissent[1]=avg_AU_per_dissent[1]/(new_dissent_level[1]-new_dissent_level[0]+1)
    avg_AU_per_dissent[2]=avg_AU_per_dissent[2]/(len(dissent)-new_dissent_level[1])


    y = []
    acc_per_dissent = [sum_true_false_dissent[0]/(sum_true_false_dissent[1]+sum_true_false_dissent[0]),
           sum_true_false_dissent[2]/(sum_true_false_dissent[3]+sum_true_false_dissent[2])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot acc per dissent
    x = [0,1]
    ax1.bar([0,1], acc_per_dissent, color='blue', label='Accuracy by number of dissenters')

    # Plot AU per dissent
    ax2.bar([0,1,2], avg_AU_per_dissent, color='red', label='Average Aleatoric Uncertainty by number of dissenters')

    ax1.locator_params(axis="x", integer=True, tight=True)
    ax2.locator_params(axis="x", integer=True, tight=True)
    #plt.title('Aleatoric Uncertainty by dissenting annotators')
    pearson = pearsonr(acc_per_dissent+[0.33], avg_AU_per_dissent)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.legend()
    plt.savefig('Figures/Aleatoric_uncertainty_by_dissent_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return avg_AU_per_dissent, acc_per_dissent

def plot_dissent_TU(pred_entropy, dissent, matcharray, params):

    dissent_asc_indeces = np.argsort(dissent)
    new_dissent_level = []

    # Where does 0 go to 1 and 1 to 2?
    for i in range(len(dissent)-1):
        if dissent[dissent_asc_indeces[i+1]] > dissent[dissent_asc_indeces[i]]:
            new_dissent_level.append(i)

    # 0: true 0 dissent, 1: false 0 dissent, 2: true 1 dissent, 3: false 1 dissent
    sum_true_false_dissent = np.zeros(4)

    for i in range(new_dissent_level[1]):
        if i <= new_dissent_level[0]:
            if matcharray[dissent_asc_indeces[i]]:
                sum_true_false_dissent[0]+=1
            else:
                sum_true_false_dissent[1]+=1
        else:
            if matcharray[dissent_asc_indeces[i]]:
                sum_true_false_dissent[2]+=1
            else:
                sum_true_false_dissent[3]+=1

    avg_AU_per_dissent = [0,0,0]

    for i in range(len(dissent)):
        if i <= new_dissent_level[0]:
            avg_AU_per_dissent[0]+=pred_entropy[dissent_asc_indeces[i]]
        elif i <= new_dissent_level[1]:
            avg_AU_per_dissent[1]+=pred_entropy[dissent_asc_indeces[i]]
        else:
            avg_AU_per_dissent[2]+=pred_entropy[dissent_asc_indeces[i]]

    avg_AU_per_dissent[0]=avg_AU_per_dissent[0]/(new_dissent_level[0]+1)
    avg_AU_per_dissent[1]=avg_AU_per_dissent[1]/(new_dissent_level[1]-new_dissent_level[0]+1)
    avg_AU_per_dissent[2]=avg_AU_per_dissent[2]/(len(dissent)-new_dissent_level[1])


    y = []
    acc_per_dissent = [sum_true_false_dissent[0]/(sum_true_false_dissent[1]+sum_true_false_dissent[0]),
           sum_true_false_dissent[2]/(sum_true_false_dissent[3]+sum_true_false_dissent[2])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot acc per dissent
    x = [0,1]
    ax1.bar([0,1], acc_per_dissent, color='blue', label='Accuracy by number of dissenters')

    # Plot AU per dissent
    ax2.bar([0,1,2], avg_AU_per_dissent, color='red', label='Average Total Uncertainty by number of dissenters')

    ax1.locator_params(axis="x", integer=True, tight=True)
    ax2.locator_params(axis="x", integer=True, tight=True)
    #plt.title('Aleatoric Uncertainty by dissenting annotators')
    pearson = pearsonr(acc_per_dissent+[0.33], avg_AU_per_dissent)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.legend()
    plt.savefig('Figures/Total_uncertainty_by_dissent_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return avg_AU_per_dissent, acc_per_dissent

def plot_dissent_EU(epis_uncertain, dissent, matcharray, params):

    dissent_asc_indeces = np.argsort(dissent)
    new_dissent_level = []

    # Where does 0 go to 1 and 1 to 2?
    for i in range(len(dissent)-1):
        if dissent[dissent_asc_indeces[i+1]] > dissent[dissent_asc_indeces[i]]:
            new_dissent_level.append(i)

    # 0: true 0 dissent, 1: false 0 dissent, 2: true 1 dissent, 3: false 1 dissent
    sum_true_false_dissent = np.zeros(4)

    for i in range(new_dissent_level[1]):
        if i <= new_dissent_level[0]:
            if matcharray[dissent_asc_indeces[i]]:
                sum_true_false_dissent[0]+=1
            else:
                sum_true_false_dissent[1]+=1
        else:
            if matcharray[dissent_asc_indeces[i]]:
                sum_true_false_dissent[2]+=1
            else:
                sum_true_false_dissent[3]+=1

    avg_AU_per_dissent = [0,0,0]

    for i in range(len(dissent)):
        if i <= new_dissent_level[0]:
            avg_AU_per_dissent[0]+=epis_uncertain[dissent_asc_indeces[i]]
        elif i <= new_dissent_level[1]:
            avg_AU_per_dissent[1]+=epis_uncertain[dissent_asc_indeces[i]]
        else:
            avg_AU_per_dissent[2]+=epis_uncertain[dissent_asc_indeces[i]]

    avg_AU_per_dissent[0]=avg_AU_per_dissent[0]/(new_dissent_level[0]+1)
    avg_AU_per_dissent[1]=avg_AU_per_dissent[1]/(new_dissent_level[1]-new_dissent_level[0]+1)
    avg_AU_per_dissent[2]=avg_AU_per_dissent[2]/(len(dissent)-new_dissent_level[1])


    y = []
    acc_per_dissent = [sum_true_false_dissent[0]/(sum_true_false_dissent[1]+sum_true_false_dissent[0]),
           sum_true_false_dissent[2]/(sum_true_false_dissent[3]+sum_true_false_dissent[2])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot acc per dissent
    x = [0,1]
    ax1.bar([0,1], acc_per_dissent, color='blue', label='Accuracy by number of dissenters')

    # Plot AU per dissent
    ax2.bar([0,1,2], avg_AU_per_dissent, color='red', label='Average Epistemic Uncertainty by number of dissenters')

    ax1.locator_params(axis="x", integer=True, tight=True)
    ax2.locator_params(axis="x", integer=True, tight=True)
    #plt.title('Aleatoric Uncertainty by dissenting annotators')
    pearson = pearsonr(acc_per_dissent+[0.33], avg_AU_per_dissent)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)))
    plt.legend()
    plt.savefig('Figures/Epistemic_uncertainty_by_dissent_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return avg_AU_per_dissent, acc_per_dissent

def plot_EU_by_target(epis_uncertainty, matcharray, true_labels, target, params):

    # if target = 0
    sum_target_0 = 0
    num_target_0 = 0
    num_correct_target_0= 0
    num_wrong_target_0 = 0

    sum_target_0_label_2= 0
    num_target_0_label_2 = 0
    num_correct_target_0_label_2= 0
    num_wrong_target_0_label_2= 0


    sum_target_1 = 0
    num_target_1 = 0
    num_correct_target_1= 0
    num_wrong_target_1 = 0

    for i in range(len(target)):
        if target[i] == 0:
            sum_target_0+= epis_uncertainty[i]
            num_target_0 += 1
            if matcharray[i] == 1:
                num_correct_target_0+=1
            else:
                num_wrong_target_0+=1

            if true_labels[i] == 2:
                sum_target_0_label_2+= epis_uncertainty[i]
                num_target_0_label_2+= 1
                if matcharray[i] == 1:
                    num_correct_target_0_label_2+=1
                else:
                    num_wrong_target_0_label_2+=1
        else:
            sum_target_1+=epis_uncertainty[i]
            num_target_1+= 1
            if matcharray[i] == 1:
                num_correct_target_1+=1
            else:
                num_wrong_target_1+=1

    acc_target_0 = num_correct_target_0 / (num_correct_target_0 + num_wrong_target_0)
    acc_target_0_label_2 = num_correct_target_0_label_2 / (num_correct_target_0_label_2 + num_wrong_target_0_label_2)
    acc_target_1 = num_correct_target_1 / (num_correct_target_1 + num_wrong_target_1)
    acc_target = (acc_target_0,acc_target_0_label_2,acc_target_1)

    avg_per_target = (sum_target_0/num_target_0,sum_target_0_label_2/num_target_0_label_2,sum_target_1/num_target_1)

    width = 0.3
    plt.bar([0,1,2], avg_per_target, width, label='Epistemic Uncertainty')
    plt.bar([0.3,1.3,2.3], acc_target, width, label='Accuracy')

    plt.title('Epistemic Uncertainty and Accuracy by test data without target group, hate speech data without target group and test data with target group', wrap=True)
    pearson = pearsonr(avg_per_target, acc_target)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)), y=-0.01)
    plt.legend()
    plt.savefig('Figures/Epistemic_uncertainty_acc_by_target_group_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return sum_target_0, sum_target_0_label_2, sum_target_1, acc_target_0,acc_target_0_label_2,acc_target_1


def plot_AU_by_target(expected_entropy, matcharray, true_labels, target, params):

    # if target = 0
    sum_target_0 = 0
    num_target_0 = 0
    num_correct_target_0= 0
    num_wrong_target_0 = 0

    sum_target_0_label_2= 0
    num_target_0_label_2 = 0
    num_correct_target_0_label_2= 0
    num_wrong_target_0_label_2= 0


    sum_target_1 = 0
    num_target_1 = 0
    num_correct_target_1= 0
    num_wrong_target_1 = 0

    for i in range(len(target)):
        if target[i] == 0:
            sum_target_0+= expected_entropy[i]
            num_target_0 += 1
            if matcharray[i] == 1:
                num_correct_target_0+=1
            else:
                num_wrong_target_0+=1

            if true_labels[i] == 2:
                sum_target_0_label_2+= expected_entropy[i]
                num_target_0_label_2+= 1
                if matcharray[i] == 1:
                    num_correct_target_0_label_2+=1
                else:
                    num_wrong_target_0_label_2+=1
        else:
            sum_target_1+=expected_entropy[i]
            num_target_1+= 1
            if matcharray[i] == 1:
                num_correct_target_1+=1
            else:
                num_wrong_target_1+=1

    acc_target_0 = num_correct_target_0 / (num_correct_target_0 + num_wrong_target_0)
    acc_target_0_label_2 = num_correct_target_0_label_2 / (num_correct_target_0_label_2 + num_wrong_target_0_label_2)
    acc_target_1 = num_correct_target_1 / (num_correct_target_1 + num_wrong_target_1)
    acc_target = (acc_target_0,acc_target_0_label_2,acc_target_1)

    avg_per_target = (sum_target_0/num_target_0,sum_target_0_label_2/num_target_0_label_2,sum_target_1/num_target_1)

    width = 0.3
    plt.bar([0,1,2], avg_per_target, width, label='Aleatoric Uncertainty')
    plt.bar([0.3,1.3,2.3], acc_target, width, label='Accuracy')

    plt.title('Aleatoric Uncertainty and Accuracy by test data without target group, hate speech data without target group and test data with target group', wrap=True)
    pearson = pearsonr(avg_per_target, acc_target)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)), y=-0.01)
    plt.legend()
    plt.savefig('Figures/Aleatoric_uncertainty_acc_by_target_group_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return sum_target_0, sum_target_0_label_2, sum_target_1, acc_target_0,acc_target_0_label_2,acc_target_1

def plot_TU_by_target(predictive_entropy, matcharray, true_labels, target, params):

    # if target = 0
    sum_target_0 = 0
    num_target_0 = 0
    num_correct_target_0= 0
    num_wrong_target_0 = 0

    sum_target_0_label_2= 0
    num_target_0_label_2 = 0
    num_correct_target_0_label_2= 0
    num_wrong_target_0_label_2= 0


    sum_target_1 = 0
    num_target_1 = 0
    num_correct_target_1= 0
    num_wrong_target_1 = 0

    for i in range(len(target)):
        if target[i] == 0:
            sum_target_0+= predictive_entropy[i]
            num_target_0 += 1
            if matcharray[i] == 1:
                num_correct_target_0+=1
            else:
                num_wrong_target_0+=1

            if true_labels[i] == 2:
                sum_target_0_label_2+= predictive_entropy[i]
                num_target_0_label_2+= 1
                if matcharray[i] == 1:
                    num_correct_target_0_label_2+=1
                else:
                    num_wrong_target_0_label_2+=1
        else:
            sum_target_1+=predictive_entropy[i]
            num_target_1+= 1
            if matcharray[i] == 1:
                num_correct_target_1+=1
            else:
                num_wrong_target_1+=1

    acc_target_0 = num_correct_target_0 / (num_correct_target_0 + num_wrong_target_0)
    acc_target_0_label_2 = num_correct_target_0_label_2 / (num_correct_target_0_label_2 + num_wrong_target_0_label_2)
    acc_target_1 = num_correct_target_1 / (num_correct_target_1 + num_wrong_target_1)
    acc_target = (acc_target_0,acc_target_0_label_2,acc_target_1)

    avg_per_target = (sum_target_0/num_target_0,sum_target_0_label_2/num_target_0_label_2,sum_target_1/num_target_1)

    width = 0.3
    plt.bar([0,1,2], avg_per_target, width, label='Total Uncertainty')
    plt.bar([0.3,1.3,2.3], acc_target, width, label='Accuracy')

    plt.title('Total Uncertainty and Accuracy by test data without target group, hate speech data without target group and test data with target group', wrap=True)
    pearson = pearsonr(avg_per_target, acc_target)
    plt.suptitle('Pearson correlation coefficient: {}, with p_value: {}'.format(round(pearson[0],3), round(pearson[1],3)), y=-0.01)
    plt.legend()
    plt.savefig('Figures/Total_uncertainty_acc_by_target_group_' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return sum_target_0, sum_target_0_label_2, sum_target_1, acc_target_0,acc_target_0_label_2,acc_target_1


def plot_highest_EU_TU_AU(epis_uncertainty,pred_entropy,exp_entropy, params):

    highest_epis_uncertainty = np.sort(epis_uncertainty)[int(len(epis_uncertainty)*0.9):]
    highest_arg_epis_uncertainty = np.argsort(epis_uncertainty)[int(len(epis_uncertainty)*0.9):]
    x = np.arange(len(highest_epis_uncertainty))
    slice_pred_entropy = np.zeros(len(highest_arg_epis_uncertainty))
    slice_exp_entropy = np.zeros(len(highest_arg_epis_uncertainty))
    sorted_pred_entropy = np.sort(pred_entropy)
    sorted_exp_entropy = np.sort(exp_entropy)

    for i in range(len(slice_pred_entropy)):
        slice_pred_entropy[i] = sorted_pred_entropy[highest_arg_epis_uncertainty[i]]
        slice_exp_entropy[i] = sorted_exp_entropy[highest_arg_epis_uncertainty[i]]

    plt.plot(x,slice_exp_entropy, label= 'Aleatoric Uncertainty of highest Epistemic Uncertainty', color='blue')
    plt.plot(x,slice_pred_entropy, label= 'Total Uncertainty of highest Epistemic Uncertainty', color='green')
    plt.title('The total and aleatoric uncertainty values for the highest 10% of epistemic uncertainty values',wrap=True)
    pearson_tu = pearsonr(x, slice_pred_entropy)
    pearson_au = pearsonr(x, slice_exp_entropy)
    plt.suptitle('Pearson correlation coefficient with TU: {}, with p_value: {}\n'.format(round(pearson_tu[0],3), round(pearson_tu[1],3))+ \
        'Pearson correlation coefficient with AU: {}, with p_value: {}\n'.format(round(pearson_au[0],3), round(pearson_au[1],3)))

    plt.legend()
    plt.savefig('Figures/Highest_EU_TU_AU_correlation' + str(params['dropout_bert']) + '_' + str(params['num_dropouts']) + get_str(params) + '.png')
    plt.close()

    return 0
