from classify import *
from drop import drop2, drop3
from fcnn import fcnn
import matplotlib.pyplot as plt

results = pd.read_csv('results.csv')

print('General results')
print('k results')
print(results.groupby(['k']).mean())
print('r results')
print(results.groupby(['r']).mean())
print('w results')
print(results.groupby(['w']).mean())
print('v results')
print(results.groupby(['v']).mean())
print('-------------------')

sat = results[results["dataset"]=='satimage']
sat_results = sat.groupby(['dataset', 'k', 'r', 'w', 'v']).mean()
print('Satimage')
print('k results')
print(sat_results.groupby(['k']).mean())
print('r results')
print(sat_results.groupby(['r']).mean())
print('w results')
print(sat_results.groupby(['w']).mean())
print('v results')
print(sat_results.groupby(['v']).mean())
print('Best combination for satimage')
best_s_row = sat_results.iloc[np.argmax(sat_results["acc"])]
print(best_s_row)
sat_acc = best_s_row["acc"]
sat_eff = best_s_row["eff"]
print('--------------')

vow = results[results["dataset"]=='vowel']
vow_results = vow.groupby(['dataset', 'k', 'r', 'w', 'v']).mean()
print('Vowel')
print('k results')
print(vow_results.groupby(['k']).mean())
print('r results')
print(vow_results.groupby(['r']).mean())
print('w results')
print(vow_results.groupby(['w']).mean())
print('v results')
print(vow_results.groupby(['v']).mean())
print('Best combination for vowel')
best_v_row = vow_results.iloc[np.argmax(vow_results["acc"])]
print(best_v_row)
vow_acc = best_v_row["acc"]
vow_eff = best_v_row["eff"]

print('Vowel menn results')
vowel_menn = pd.read_csv('vowel_menn.csv')
vowel_menn = vowel_menn.groupby(['k']).mean()
print(vowel_menn)
best_v_menn_row= vowel_menn.iloc[np.argmax(vowel_menn["acc"])]
print(best_v_menn_row)
print('Accuracy ratio: ', best_v_menn_row["acc"]/vow_acc)
print('Efficiency ratio: ', best_v_menn_row["eff"]/vow_eff)
print('Storage ratio: ', best_v_menn_row["storage"])

print('Satimage menn results')
satimage_menn = pd.read_csv('satimage_menn.csv')
satimage_menn = satimage_menn.groupby(['k']).mean()
print(satimage_menn)
best_s_menn_row= satimage_menn.iloc[np.argmax(satimage_menn["acc"])]
print(best_s_menn_row)
print('Accuracy ratio: ', best_s_menn_row["acc"]/sat_acc)
print('Efficiency ratio: ', best_s_menn_row["eff"]/sat_eff)
print('Storage ratio: ', best_s_menn_row["storage"])

print('Vowel fcnn results')
vowel_fcnn = pd.read_csv('vowel_fcnn.csv')
best_v_fcnn_row= vowel_fcnn.mean()
print(best_v_fcnn_row)
print('Accuracy ratio: ', best_v_fcnn_row["acc"]/vow_acc)
print('Efficiency ratio: ', best_v_fcnn_row["eff"]/vow_eff)
print('Storage ratio: ', best_v_fcnn_row["storage"])

print('Satimage fcnn results')
satimage_fcnn = pd.read_csv('satimage_fcnn.csv')
best_s_fcnn_row = satimage_fcnn.mean()
print(best_s_fcnn_row)
print('Accuracy ratio: ', best_s_fcnn_row["acc"]/sat_acc)
print('Efficiency ratio: ', best_s_fcnn_row["eff"]/sat_eff)
print('Storage ratio: ', best_s_fcnn_row["storage"])

print('Vowel drop3 results')
vowel_drop3 = pd.read_csv('vowel_drop3.csv')
best_v_drop3_row = vowel_drop3.mean()
print(best_v_drop3_row)
print('Accuracy ratio: ', best_v_drop3_row["acc"]/vow_acc)
print('Efficiency ratio: ', best_v_drop3_row["eff"]/vow_eff)
print('Storage ratio: ', best_v_drop3_row["storage"])

print('Satimage drop3 results')
satimage_drop3 = pd.read_csv('satimage_drop3.csv')
best_s_drop3_row = satimage_drop3.mean()
print(best_s_drop3_row)
print('Accuracy ratio: ', best_s_drop3_row["acc"]/sat_acc)
print('Efficiency ratio: ', best_s_drop3_row["eff"]/sat_eff)
print('Storage ratio: ', best_s_drop3_row["storage"])

drop3_v_values = [best_v_drop3_row["storage"], best_v_drop3_row["eff"]/vow_eff, best_v_drop3_row["acc"]]
menn_v_values = [best_v_menn_row["storage"], best_v_menn_row["eff"]/vow_eff, best_v_menn_row["acc"]]
fcnn_v_values = [best_v_fcnn_row["storage"], best_v_fcnn_row["eff"]/vow_eff, best_v_fcnn_row["acc"]]


width = 0.8
indices = [1, 3, 5]
plt.title('Vowel dataset results')
plt.bar(indices, [1, 1, vow_acc], width=width, color='black', label='Without reduction')
plt.bar([i+0.25*width for i in indices], menn_v_values, width=width, color='slateblue', label='Menn')
plt.bar([i+0.5*width for i in indices], fcnn_v_values, width=width, color='firebrick', label='Fcnn')
plt.bar([i+0.75*width for i in indices], drop3_v_values, width=width, color='limegreen', label='Drop3')

plt.xticks(indices, ['Storage', 'Time', 'Accuracy'])
plt.legend()
plt.show()

drop3_s_values = [best_s_drop3_row["storage"], best_s_drop3_row["eff"]/sat_eff, best_s_drop3_row["acc"]]
menn_s_values = [best_s_menn_row["storage"], best_s_menn_row["eff"]/sat_eff, best_s_menn_row["acc"]]
fcnn_s_values = [best_s_fcnn_row["storage"], best_s_fcnn_row["eff"]/sat_eff, best_s_fcnn_row["acc"]]

width = 0.8
indices = [1, 3, 5]
plt.title('Satimage dataset results')
plt.bar(indices, [1, 1, sat_acc], width=width, color='black', label='Without reduction')
plt.bar([i+0.25*width for i in indices], menn_s_values, width=width, color='slateblue', label='Menn')
plt.bar([i+0.5*width for i in indices], fcnn_s_values, width=width, color='firebrick', label='Fcnn')
plt.bar([i+0.75*width for i in indices], drop3_s_values, width=width, color='limegreen', label='Drop3')

plt.xticks(indices, ['Storage', 'Time', 'Accuracy'])
plt.legend()
plt.show()