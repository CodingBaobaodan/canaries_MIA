import numpy
import matplotlib.pyplot as plt
from utils import calibrate_logits,cal_results_jilin

#pred = numpy.load('/home/915688516/code/canary_main/canary/result/jilin_net/jilin_net.npz')
#inf = numpy.load('/home/915688516/code/canary_main/canary/result/jilin_net/inf_score.npy.npz')
#result = numpy.load('/home/915688516/code/canary_main/canary/loss_result.npy.npz') # vulnerable datapoint loss_diff result from non def model, --logit 40 0, --num 30 --iter 5000 --compare 10
#result = numpy.load('/home/915688516/code/canary_main/canary/loss_result_2.npy.npz') # vulnerable datapoint loss_diff result from def model, --logit 10 0, --num 10 --iter 500 -- compare 10
result = numpy.load('/home/915688516/code/canary_main/canary/loss_each_canary.npy.npz') # vulnerable datapoint loss result for each canary from def model, --logit 10 0, --num 10 --iter 500 -- compare 10

'''
# load out data 
pred_logits = pred['pred_logits']
in_out_labels = pred['in_out_labels']
canary_losses = pred['canary_losses']
class_labels = pred['class_labels']
img_id = pred['img_id']
#all_canaries = pred['all_canaries']
'''
'''
loss_diff_list = []
id_list = []
loss_diff_list = [result['arr_0'][i]['loss'] for i in range(len(result['arr_0']))]
id_list = [result['arr_0'][i]['id'] for i in range(len(result['arr_0']))]
print(f"id : {id_list}")
'''
result_dict = result['arr_0']
fig, axs = plt.subplots(2,3)
# Plot the histograms on each subplot
axs[0][0].hist(result_dict[0]['loss'], bins='auto')
axs[0][1].hist(result_dict[1]['loss'], bins='auto')
axs[0][2].hist(result_dict[2]['loss'], bins='auto')
axs[1][0].hist(result_dict[3]['loss'], bins='auto')
axs[1][1].hist(result_dict[4]['loss'], bins='auto')
axs[1][2].hist(result_dict[5]['loss'], bins='auto')


#print(f"len of loss list : {len(loss_diff_list)}")
#plt.hist(loss_diff_list, bins=5)
plt.title('Histogram of diff')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.savefig('/home/915688516/code/canary_main/canary/diff.png')

'''
# the diff stores a list of dictionary containing id and it's corresponding diff value 
# Note: # of diff value for one id depends on num_gen
print(diff)
print(f"diff length {len(diff['arr_0'])}")
print(f"diff {diff['arr_0'][0]['diff']}")


# standard calculation of output stats
in_out_labels = numpy.swapaxes(in_out_labels, 0, 1).astype(bool)
pred_logits = numpy.swapaxes(pred_logits, 0, 1)
scores = calibrate_logits(pred_logits, class_labels, 'log_logits')
shadow_scores = scores[:-1]
target_scores = scores[-1:]
shadow_in_out_labels = in_out_labels[:-1]
target_in_out_labels = in_out_labels[-1:]
some_stats = cal_results_jilin(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=1)
'''












'''
# select out the vulnerable datapoints
threshold_low = some_stats['fix_threshold@0.001FPR'] 
vulnerable_datapoint_id = (numpy.argwhere(some_stats['fix_final_preds'] >= threshold_low)).flatten()
selected_canaries = []
for i in vulnerable_datapoint_id:
    selected_canaries.append(all_canaries[i])
'''
#print(f"threshold value at 0.001 FPR is: {threshold_low}")
#print(f"vulnerable datapoint ids are: {vulnerable_datapoint_id}")
#print(f"selected canaries' length: {len(selected_canaries)}")
#print(f"one canary shape: {selected_canaries[0].shape}")


'''
vulnerable_datapoint_inf = []
for id in vulnerable_datapoint_id:
    vulnerable_datapoint_inf.append(inf['arr_0'][id])

#print(numpy.squeeze(vulnerable_datapoint_inf[1]))

plt.hist(numpy.squeeze(vulnerable_datapoint_inf[1]), bins='auto')
plt.title('Histogram of influence score')
plt.xlabel('Influence')
plt.ylabel('Frequency')
plt.show()
plt.savefig('/home/915688516/code/canary_main/canary/influence_image.png')



#print(some_stats['fix_threshold'])
#print(some_stats['fix_TPR_tpr'])
#print(some_stats['fix_TPR_fpr'])
#print("the threshold is at below:")
#print(some_stats['fix_threshold@0.001FPR'])
#print(scores.shape)

'''

