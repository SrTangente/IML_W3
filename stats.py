import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare as friedman
from scipy.stats import f_oneway as anova
results = pd.read_csv('results.csv')


def stats():
    sat_k = results[(results["dataset"]=='satimage') &  (results["r"]==2) & (results["v"]=='maj') & (results["w"]=='eq')]
    sat_k_results = sat_k.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    sat_r = results[(results["dataset"]=='satimage') & (results["k"]==3)& (results["v"]=='maj') & (results["w"]=='eq')]
    sat_r_results = sat_r.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    sat_v = results[(results["dataset"]=='satimage') & (results["k"]==3) & (results["r"]==2) &  (results["w"]=='eq')]
    sat_v_results = sat_v.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    sat_w = results[(results["dataset"]=='satimage') & (results["k"]==3)& (results["r"]==2) & (results["v"]=='maj') ]
    sat_w_results = sat_w.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    print('k results')
    print(sat_k_results['acc'].max())
    print(sat_k_results.iloc[np.argmax(sat_k_results["acc"])])
    print('r results')
    print(sat_r_results.iloc[np.argmax(sat_r_results["acc"])])
    print(sat_r_results['acc'].max())
    print('v results')
    print(sat_v_results.iloc[np.argmax(sat_v_results["acc"])])
    print(sat_v_results['acc'].max())
    print('w results')
    print(sat_w_results.iloc[np.argmax(sat_w_results["acc"])])
    print(sat_w_results['acc'].max())


    vowel_k = results[(results["dataset"]=='vowel') &  (results["r"]==2) & (results["v"]=='maj') & (results["w"]=='eq')]
    vowel_k_results = vowel_k.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    vowel_r = results[(results["dataset"]=='vowel') & (results["k"]==3)& (results["v"]=='maj') & (results["w"]=='eq')]
    vowel_r_results = vowel_r.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    vowel_v = results[(results["dataset"]=='vowel') & (results["k"]==3) & (results["r"]==2) &  (results["w"]=='eq')]
    vowel_v_results = vowel_v.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    vowel_w = results[(results["dataset"]=='vowel') & (results["k"]==3)& (results["r"]==2) & (results["v"]=='maj') ]
    vowel_w_results = vowel_w.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()

    print('k results')
    print(vowel_k_results['acc'].max())
    print(vowel_k_results.iloc[np.argmax(vowel_k_results["acc"])])
    print('r results')
    print(vowel_r_results.iloc[np.argmax(vowel_r_results["acc"])])
    print(vowel_r_results['acc'].max())
    print('v results')
    print(vowel_v_results.iloc[np.argmax(vowel_v_results["acc"])])
    print(vowel_v_results['acc'].max())
    print('w results')
    print(vowel_w_results.iloc[np.argmax(vowel_w_results["acc"])])
    print(vowel_w_results['acc'].max())

    sat = results[results["dataset"]=='satimage']
    sat_results = sat.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()

    print('Best combination for satimage')
    print(sat_results.iloc[np.argmax(sat_results["acc"])])
    print('--------------')

    vowel = results[results["dataset"]=='vowel']
    vowel_results = vowel.groupby(['dataset', 'k', 'r', 'w', 'v'],as_index=False).mean()
    print('Best combination for vowel')
    print(vowel_results.iloc[np.argmax(vowel_results["acc"])])


    sat_best_k=sat_k_results.iloc[np.argmax(sat_k_results["acc"])]
    sat_best_r=sat_r_results.iloc[np.argmax(sat_r_results["acc"])]
    sat_best_v=sat_v_results.iloc[np.argmax(sat_v_results["acc"])]
    sat_best_w=sat_w_results.iloc[np.argmax(sat_w_results["acc"])]
    sat_best=sat_results.iloc[np.argmax(sat_results["acc"])]

    sat_folds_best_k=sat[(sat['k']==sat_best_k['k'])&(sat['r']==sat_best_k['r'])&(sat['v']==sat_best_k['v'])&(sat['w']==sat_best_k['w'])]['acc']
    sat_folds_best_r=sat[(sat['k']==sat_best_r['k'])&(sat['r']==sat_best_r['r'])&(sat['v']==sat_best_r['v'])&(sat['w']==sat_best_r['w'])]['acc']
    sat_folds_best_v=sat[(sat['k']==sat_best_v['k'])&(sat['r']==sat_best_v['r'])&(sat['v']==sat_best_v['v'])&(sat['w']==sat_best_v['w'])]['acc']
    sat_folds_best_w=sat[(sat['k']==sat_best_w['k'])&(sat['r']==sat_best_w['r'])&(sat['v']==sat_best_w['v'])&(sat['w']==sat_best_w['w'])]['acc']
    sat_folds_best=sat[(sat['k']==sat_best['k'])&(sat['r']==sat_best['r'])&(sat['v']==sat_best['v'])&(sat['w']==sat_best['w'])]['acc']

    vowel_best_k=vowel_k_results.iloc[np.argmax(vowel_k_results["acc"])]
    vowel_best_r=vowel_r_results.iloc[np.argmax(vowel_r_results["acc"])]
    vowel_best_v=vowel_v_results.iloc[np.argmax(vowel_v_results["acc"])]
    vowel_best_w=vowel_w_results.iloc[np.argmax(vowel_w_results["acc"])]
    vowel_best=vowel_results.iloc[np.argmax(vowel_results["acc"])]

    vowel_folds_best_k=vowel[(vowel['k']==vowel_best_k['k'])&(vowel['r']==vowel_best_k['r'])&(vowel['v']==vowel_best_k['v'])&(vowel['w']==vowel_best_k['w'])]['acc']
    vowel_folds_best_r=vowel[(vowel['k']==vowel_best_r['k'])&(vowel['r']==vowel_best_r['r'])&(vowel['v']==vowel_best_r['v'])&(vowel['w']==vowel_best_r['w'])]['acc']
    vowel_folds_best_v=vowel[(vowel['k']==vowel_best_v['k'])&(vowel['r']==vowel_best_v['r'])&(vowel['v']==vowel_best_v['v'])&(vowel['w']==vowel_best_v['w'])]['acc']
    vowel_folds_best_w=vowel[(vowel['k']==vowel_best_w['k'])&(vowel['r']==vowel_best_w['r'])&(vowel['v']==vowel_best_w['v'])&(vowel['w']==vowel_best_w['w'])]['acc']
    vowel_folds_best=vowel[(vowel['k']==vowel_best['k'])&(vowel['r']==vowel_best['r'])&(vowel['v']==vowel_best['v'])&(vowel['w']==vowel_best['w'])]['acc']


    print(friedman(vowel_folds_best_k,vowel_folds_best_r,vowel_folds_best_v,vowel_folds_best_w,vowel_folds_best))
    print(friedman(sat_folds_best_k,sat_folds_best_r,sat_folds_best_v,sat_folds_best_w,sat_folds_best))
    print(anova(vowel_folds_best_k,vowel_folds_best_r,vowel_folds_best_v,vowel_folds_best_w,vowel_folds_best))
    print(anova(sat_folds_best_k,sat_folds_best_r,sat_folds_best_v,sat_folds_best_w,sat_folds_best))