# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:47:03 2015

@author: akkus
"""
import os, string
import random
import pandas as pd
import numpy as np
import statsmodels as sm
import hddm
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from scipy import stats
### Importing R objects
from rpy2 import robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.lib.ggplot2 as gg
from rpy2.robjects.packages import importr
base = importr('base')
rstats = importr('stats')
ez = importr('ez')
ggplot2 = importr('ggplot2')

def collect_data(path, ext='csv', delim=','):
    data_list = []
    dir_list = os.listdir(path)
    for i, dir_file in enumerate(dir_list):
        if dir_file.endswith('.%s' % ext):
            single_data = pd.read_csv('%s/%s' % (path, dir_file), sep=delim, names=list(string.ascii_uppercase))
            data_list.insert(i, single_data)
    data = pd.concat(data_list, ignore_index=True)
    return data

def clean_data(data, sort_col, sort_val, header={}):
    cleaned = data[data[sort_col] == sort_val]
    cleaned = cleaned.dropna(axis=1,how='all')
    cleaned = cleaned.convert_objects(convert_numeric=True)
    cleaned = cleaned.reset_index()
    cleaned = cleaned.drop('index', axis=1)
    cleaned = cleaned.rename(columns = header)
    return cleaned

def participant_list(data, id_name='ID'):
    n = 0
    sub_list = [data.iloc[0][id_name]]
    prev_id = sub_list[0]
    for i, current_id in enumerate(data[id_name]):
        if current_id != prev_id:
            n = n+1
            sub_list.insert(n, current_id)
        prev_id = current_id
    sub_list.sort()
    return sub_list

def count_trial(data, sub_list=None, id_name='ID', cond_name='Resp_code', cond_inst=['L_t1', 'R_t1'], value='Resp_key'):
    C_trial = pd.pivot_table(data, index=[id_name], columns=[cond_name], values=[value], aggfunc=[np.count_nonzero])
    C_trial = pd.DataFrame(C_trial.values, columns=cond_inst)
    if sub_list: C_trial.index = sub_list
    else: C_trial.index = sorted(list(set(data[id_name])))
    return C_trial

def error_rates(data, sub_list=None, id_name='ID', err_name='Error'):
    E_rate = pd.pivot_table(data, index=[id_name], values=[err_name], aggfunc=[np.mean])
    E_rate = pd.DataFrame((E_rate.values*100), columns=['Mean error (%)'])
    if sub_list: E_rate.index = sub_list
    else: E_rate.index = sorted(list(set(data[id_name])))
    return E_rate

def filt_perc(data, var, low='2.5%', high='97.5%'):
    desc_data = data[var].describe(percentiles=[.01, .025, .05, .125, .25, .5, .75, .875, .95, .975, .99])
    data = data[data[var] >= desc_data[low]]
    data = data[data[var] <= desc_data[high]]
    return data
    
def filt_sd(data, dep_var, id_var, scope=2):
    filt_data = pd.DataFrame()
    for sub in list(set(data[id_var])):
        sub_data = data[data[id_var] == sub]
        desc = sub_data[dep_var].describe()
        sub_data = sub_data[sub_data[dep_var] >= desc['mean']-scope*desc['std']]
        sub_data = sub_data[sub_data[dep_var] <= desc['mean']+scope*desc['std']]
        filt_data = pd.concat([filt_data, sub_data])
    return filt_data

def revert_error(data, err_name='Error'):
    correct = []
    for i, err in enumerate(data[err_name]):
        if err == 0:
            correct.insert(i, 1)
        elif err == 1:
            correct.insert(i, 0)
    data_corr = data.assign(Correct = correct)
    return data_corr

def linreg_table(data, sub_list=None, predictor=range(1,10), concat=True):
    linreg_lab = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']
    X = predictor
    data_linreg = pd.DataFrame(columns = linreg_lab)
    for i, y in enumerate(data.iterrows()):
        Y = []
        for j, x in enumerate(predictor):
            Y.insert(j, data.values[i][j])
        slope = pd.DataFrame([stats.linregress(X,Y)], columns = linreg_lab)
        data_linreg = pd.concat([data_linreg, slope], ignore_index=True)
    if sub_list: 
        data_linreg.index = sub_list
        data.index = sub_list
    else: data_linreg.index = data.index
    
    if concat == True:      
          data_full = pd.concat([data, data_linreg,], axis=1)
    else:
          data_full = data_linreg
    return data_full

def get_ez_params (Pc, VRT, MRT, s=0.1):
    # TODO rename function and var names; comments for the variables
    """ Based on the R code published in
    Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007).
    An EZ-diffusion model for response time and accuracy.
    Psychonomic Bulletin & Review, 14(1), 3–22.
    
    Parameters:
        pc: percent of correct responses
        vrt: variance of the correct response times
        mrt: mean of the correct response times
        s: drift variance or scaling parameter(default: 0.1)
    Returns:
        v: drift rate
        a: difference between thresholds
        ter: nondecision time
    """
    s2 = s**2
    if Pc == 0:
        print 'Oops, Pc == 0!'
    elif Pc == 0.5:
        print 'Oops, Pc == .5!'
    elif Pc == 1:
        print 'Oops, Pc == 1!'
    # If Pc equals 0, .5, or 1, the method will not work, and an edge correction is required.
    L = np.log(Pc/(1-Pc))  # The original R function “qlogis” calculates the logit.
    x = L*(L*Pc**2 - L*Pc + Pc - .5)/VRT
    v = np.sign(Pc-.5)*s*x**(0.25)  # This gives drift rate.
    a = s2*np.log(Pc/(1-Pc))/v  # This gives boundary separation.
    y = -v*a/s2
    MDT = (a/(2*v)) * (1-np.exp(y))/(1+np.exp(y))
    ter = MRT - MDT  # This gives nondecision time.
    return v, a, ter

def EZ_table(pc_table, rtm_table, rtv_table, col_names, sub_list, cell_size, hdd=False):
    if hdd: 
        ez_func = hddm.utils.EZ
    else:
        ez_func = get_ez_params
    table_v = pd.DataFrame(columns = col_names)
    table_a = pd.DataFrame(columns = col_names)
    table_t = pd.DataFrame(columns = col_names)
    for i, pc_row in enumerate(pc_table.iterrows()):
        v_row = []
        a_row = []
        t_row = []
        for j, pc_loc in enumerate(pc_row[1]):
            #Edge-correction
            if pc_loc == 1.0:
                pc_loc = 1-1.0/(cell_size*2)
            ez = get_ez_params(pc_loc, rtv_table.values[i][j], rtm_table.values[i][j])
            v_row.insert(j, ez[0])
            a_row.insert(j, ez[1])
            t_row.insert(j, ez[2])
        v_row_df = pd.DataFrame([v_row], columns = col_names)
        a_row_df = pd.DataFrame([a_row], columns = col_names)
        t_row_df = pd.DataFrame([t_row], columns = col_names)
        table_v = pd.concat([table_v, v_row_df], ignore_index=True)
        table_a = pd.concat([table_a, a_row_df], ignore_index=True)
        table_t = pd.concat([table_t, t_row_df], ignore_index=True)    
    table_v.index = sub_list
    table_a.index = sub_list
    table_t.index = sub_list
    return table_v, table_a, table_t

def EZ_skewness(data, ri='RTms', indep_var='Congruency', id_var='ID', seq=['C', 'N', 'I'], snarc=True):
    mean_table = pd.pivot_table(data, values=ri, index=id_var, columns=indep_var, aggfunc=np.mean)
    mean_table = pd.DataFrame(mean_table.values, index=sorted(set(data[id_var])), columns=sorted(seq))
    mean_table = mean_table[seq]
    if snarc: mean_table = mean_table.rename(columns=dict(zip(seq,range(len(seq)))))
    skew_table = pd.pivot_table(data, values=ri, index=id_var, columns=indep_var, aggfunc=stats.skew)
    skew_table = pd.DataFrame(skew_table.values, index=sorted(set(data[id_var])), columns=sorted(seq))
    skew_table = skew_table[seq]
    if snarc: skew_table = skew_table.rename(columns=dict(zip(seq,range(len(seq)))))
    linreg_table = pd.DataFrame(columns=['slope', 'intercept', 'r_value', 'p_value', 'std_err'])
    for i, sub in enumerate(mean_table.iterrows()):
        y=[]
        x=[]
        for j, cond in enumerate(list(set(data[indep_var]))):
            x.append(mean_table.iloc[i][j])
            y.append(skew_table.iloc[i][j])
        linreg_row = pd.DataFrame([stats.linregress(x,y)], columns=['slope', 'intercept', 'r_value', 'p_value', 'std_err'])
        linreg_table = pd.concat([linreg_table, linreg_row])
    linreg_table.index = sorted(list(set(data[id_var])))
    return linreg_table

def EZ_errorspeed2(data, ri='RTms', id_var='ID', error='Error'):
    error_speed = pd.pivot_table(data, values=ri, index=id_var, columns=error, aggfunc=np.mean)
    error_speed = pd.DataFrame(error_speed.values, index=sorted(list(set(data[id_var]))), columns=['Accurate', 'Erroneous'])
    error_speed = error_speed.dropna(axis=0, how='any')
    return error_speed
    
def EZ_errorspeed(data, ri='RTms', indep_var='Congruency', id_var='ID', error='Error'):
    corrects = data[data[error]==0]
    errors = data[data[error]==1]
    # Mean of correct RTs
    correct_table = pd.pivot_table(corrects, values=ri, index=id_var, columns=indep_var, aggfunc=np.mean)
    correct_table = pd.DataFrame(correct_table.values, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[indep_var]))))
    corr_means = []
    for sub in sorted(list(set(data[id_var]))):
        corr_means.append(np.mean(correct_table.loc[sub]))
    corr_fill=np.empty((0,len(corr_means)))
    for i in range(len(set(data[indep_var]))):
        corr_fill=np.vstack((corr_fill, np.asarray(corr_means)))
    corr_fill = np.transpose(corr_fill)
    corr_fill = pd.DataFrame(corr_fill, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[indep_var]))))
    correct_table = correct_table.fillna(value=corr_fill)
    # Mean of erroneous RTs
    error_table = pd.pivot_table(errors, values=ri, index=id_var, columns=indep_var, aggfunc=np.mean)
    error_table = pd.DataFrame(error_table.values, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[indep_var]))))
    err_means = []
    for sub in sorted(list(set(data[id_var]))):
        err_means.append(np.mean(error_table.loc[sub]))
    err_fill=np.empty((0,len(err_means)))
    for i in range(len(set(data[indep_var]))):
        err_fill=np.vstack((err_fill, np.asarray(err_means)))
    err_fill = np.transpose(err_fill)
    err_fill = pd.DataFrame(err_fill, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[indep_var]))))
    error_table = error_table.fillna(value=err_fill)
    # Difference
    diff_table = pd.DataFrame(correct_table.values - error_table.values, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[indep_var]))))
    return diff_table

def EZ_bias(data, ri='RTms', id_var='ID', stim_var='t_color', error='Error'):
    errors = data[data['Error']==1]
    corrects = data[data['Error']==0]
    # Erroneous RTs
    error_table = pd.pivot_table(errors, values=ri, index=id_var, columns=stim_var, aggfunc=np.mean)
    error_table = pd.DataFrame(error_table.values, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[stim_var]))))
    err_means = []
    for sub in sorted(list(set(data[id_var]))):
        err_means.append(np.mean(error_table.loc[sub]))
    err_fill=np.empty((0,len(err_means)))
    for i in range(len(set(data[stim_var]))):
        err_fill=np.vstack((err_fill, np.asarray(err_means)))
    err_fill = np.transpose(err_fill)
    err_fill = pd.DataFrame(err_fill, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[stim_var]))))
    error_table = error_table.fillna(value=err_fill)
    # Correct RTs
    correct_table = pd.pivot_table(corrects, values=ri, index=id_var, columns=stim_var, aggfunc=np.mean)
    correct_table = pd.DataFrame(correct_table.values, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[stim_var]))))
    corr_means = []
    for sub in sorted(list(set(data[id_var]))):
        corr_means.append(np.mean(correct_table.loc[sub]))
    corr_fill=np.empty((0,len(corr_means)))
    for i in range(len(set(data[stim_var]))):
        corr_fill=np.vstack((corr_fill, np.asarray(corr_means)))
    corr_fill = np.transpose(corr_fill)
    corr_fill = pd.DataFrame(corr_fill, index=sorted(list(set(data[id_var]))), columns=sorted(list(set(data[stim_var]))))
    correct_table = correct_table.fillna(value=corr_fill)
    # Bias
    bias_table = correct_table.subtract(error_table).values
    bias_table = pd.DataFrame(bias_table, index=sorted(list(set(data[id_var]))), columns=['t1','t2'])
    return bias_table

### ANOVA
def rm_ANOVA(data, dep_var, indep_var=None, id_var=None, wide=True):
    """
    Standard one-way repeated measures ANOVA
    
    ### Arguments:
    data: pandas DataFrame
    dep_var: dependent variable - label (long format) or a list of labels (wide format)
    indep_var: label of the independent variable (only necessary if data is in long format)
    id_var: label of the variable which contains the participants' identifiers. Default assumes that the table index contains the identifiers.
    wide: whether the data is in wide format
    
    ### Returns: [DFn, DFd, F, pF, W, pW], corr_table
    DFn, DFd: degrees of freedom used when calculating p(F)
    F: F statistic (uncorrected)
    pF: p value of the F statistic (uncorrected)
    W: statistic of Mauchly's test for sphericity
    pW: p value of Mauchly's test for sphericity (sphericity is violated if p(W)<0.05)
    corr_table: numpy array, contains Greenhouse & Geisser's, Huhyn & Feldt's, and the "lower-bound" epsilons, and the corrected p values of F
    """
    ### Reshaping data
    if wide:
        if not id_var:
            data = data.assign(ID=data.index)
            id_var = 'ID'
        data = pd.melt(data, id_vars=id_var, value_vars=dep_var, var_name='condition', value_name='measured')
        dep_var = 'measured'
        indep_var = 'condition'
    ### one-way ANOVA
    n = len(set(data[id_var]))
    k = len(set(data[indep_var]))
    DFn = (k-1)
    DFd = (k-1)*(n-1)
    q_eff = []
    q_err = []
    for j, var in enumerate(list(set(data[indep_var]))):
        subset_j = data[data[indep_var] == var]
        q_eff.insert(j, n*np.square(np.mean(subset_j[dep_var])-np.mean(data[dep_var])))
        for i, sub in enumerate(list(set(data[id_var]))):
            subset_i = data[data[id_var] == sub]
            subset_ij = subset_j[subset_j[id_var] == sub]
            q_err.insert(j+i, np.square(np.mean(subset_ij[dep_var]) -np.mean(subset_i[dep_var]) - np.mean(subset_j[dep_var]) + np.mean(data[dep_var])))
    # F-statistic    
    F = (sum(q_eff)/DFn)/(sum(q_err)/DFd)
    pF = 1-stats.f.cdf(F, DFn, DFd)
    ### Mauchly's test for sphericity & Degree of freedom corrections
    # Calculating sample covariances
    table = np.empty((0,n))
    for j, var in enumerate(list(set(data[indep_var]))):
        subset_j = data[data[indep_var] == var]
        row = []
        for i, sub in enumerate(list(set(data[id_var]))):
            subset_ij = subset_j[subset_j[id_var] == sub]
            row.insert(i, np.mean(subset_ij[dep_var].values))
        table = np.vstack([table, np.asarray(row)])    
    samp_table = np.cov(table)
    samp_means = samp_table.mean(axis=1)
    # Estimating population covariances
    pop_table = np.empty((0,k))
    for x in range(k):
        row = []
        for y in range(k):
            row.insert(y, samp_table[x][y]-samp_means[x]-samp_means[y]+samp_table.mean())
        pop_table = np.vstack([pop_table, np.asarray(row)])
    # Mauchly's W statistic
    W = np.prod([x for x in list(np.linalg.eigvals(pop_table)) if x > 0.00000000001])/np.power(np.trace(pop_table)/(k-1), (k-1)) # uses the pseudo-determinant (discards all near-zero eigenvalues)
    dfW = int((0.5*k*(k-1))-1)
    fW = float(2*np.square(k-1)+(k-1)+2)/float(6*(k-1)*(n-1))
    chiW = (fW-1)*(n-1)*np.log(W)
    pW = 1-stats.chi2.cdf(chiW, dfW)
    # Greenhouse & Geisser's epsilon   
    GG = np.square(np.trace(pop_table))/(np.sum(np.square(pop_table))*(k-1))
    # Huynh & Feldt's epsilon
    HF = (n*(k-1)*GG-2)/((k-1)*(n-1-(k-1)*GG))
    # Lower-bound epsilon
    LB = 1/float(k-1)
    # Correction
    corr_list = [GG,HF,LB]
    corr_table = np.empty((0,2))
    for epsilon in corr_list:
        F_corr = (sum(q_eff)/(DFn*epsilon))/(sum(q_err)/(DFd*epsilon))
        pF_corr = 1-stats.f.cdf(F_corr, DFn*epsilon, DFd*epsilon)
        corr_table = np.vstack([corr_table, np.array([epsilon, pF_corr])])
    return [DFn,DFd,F,pF,W,pW], corr_table

def pairwise_ttest(data, dep_var, indep_var=None, id_var=None, wide=True):
    ### Reshaping data
    if wide:
        if not id_var:
            data = data.assign(ID=data.index)
            id_var = 'ID'
        data = pd.melt(data, id_vars=id_var, value_vars=dep_var, var_name='condition', value_name='measured')
        dep_var = 'measured'
        indep_var = 'condition'    
    # Pairwise t-tests
    table = np.empty((0,2))
    pairings = []
    for var in list(set(data[indep_var])):
        for var2 in list(set(data[indep_var])):
            if var != var2 and '%s - %s'%(var2,var) not in pairings:
                subset_var = data[data[indep_var]==var]
                subset_var2 = data[data[indep_var]==var2]
                table = np.vstack([table, np.asarray(stats.ttest_rel(subset_var[dep_var], subset_var2[dep_var]))])
                pairings.append('%s - %s'%(var,var2))
    # Corrections
    fam_size = (np.square(len(set(data[indep_var])))-len(set(data[indep_var])))/2
    bonf_list = []
    holm_list = []
    sorted_p = sorted(list(table[:,1]))
    for p in table[:,1]:
        p_bonf = p*fam_size
        p_holm = p*(fam_size-sorted_p.index(p))
        if p_bonf > 1: p_bonf = 1
        if p_holm > 1: p_holm = 1
        bonf_list.append(p_bonf)
        holm_list.append(p_holm)
    table = np.hstack([table, np.asarray(zip(bonf_list, holm_list))])
    table = pd.DataFrame(table, index=pairings, columns=['t','p','p (Bonf)','p (Holm)'])
    return table
    
def oneway_repANOVA(variables, sub_list, dep_name='measured', indep_name='condition'):
    ### converting passed variables to long format
    data = pd.concat(variables, axis=1,)
    data.index = range(len(variables[0]))
    id_var = pd.DataFrame(sub_list, columns=['ID'])
    data = pd.concat([id_var, data], axis=1,)
    labels = []
    for i, var in enumerate(variables):
        labels.insert(i, var.name)
    data = pd.melt(data, id_vars='ID', value_vars=labels, var_name=indep_name, value_name=dep_name)
    ### Testing
    print rm_ANOVA(data, dep_name, indep_name, 'ID', wide=False)
    print pairwise_ttest(data, dep_name, indep_name, 'ID', wide=False)
    #print sm.pairwise_tukeyhsd(data[dep_name], data[indep_name], alpha=0.05)
    data.boxplot(by=indep_name)
    data = pandas2ri.py2ri(data)
    ### using ezANOVA function from R
    model = ez.ezANOVA(data=data, dv=base.as_symbol(dep_name), wid=base.as_symbol('ID'), within=base.as_symbol(indep_name), type=3)
    print model
    ### post hoc tests
    posthoc = rstats.pairwise_t_test(data.rx2(dep_name), data.rx2(indep_name), paired = True, p_adjust_method = "bonferroni")
    print 'Pairwise t tests (Bonferroni corrected)'
    print posthoc[2]

def pyezANOVA(data,dep_name,indep_name,id_name):
    data = pandas2ri.py2ri(data)
    model = ez.ezANOVA(data=data, dv=base.as_symbol(dep_name), wid=base.as_symbol(id_name), within=base.as_symbol(indep_name), type=3)
    print model
    
def py_posthoc(data,dep_name,indep_name,id_name):
    data = pandas2ri.py2ri(data)
    posthoc = rstats.pairwise_t_test(data.rx2(dep_name), data.rx2(indep_name), paired = True, p_adjust_method = "bonferroni")
    print posthoc[2]
    
def bar_plot(variables, sub_list, dep_name='measured', indep_name='condition',):
    ### converting passed variables to long format
    data = pd.concat(variables, axis=1,)
    data.index = range(len(variables[0]))
    id_var = pd.DataFrame(sub_list, columns=['ID'])
    data = pd.concat([id_var, data], axis=1,)
    labels = []
    for i, var in enumerate(variables):
        labels.insert(i, var.name)
    data = pd.melt(data, id_vars='ID', value_vars=labels, var_name=indep_name, value_name=dep_name)
    data = pandas2ri.py2ri(data)
    ### using ggplot2 module from R
    plt = gg.ggplot(data)
    plt = plt + \
    gg.aes_string(x=indep_name, y=dep_name) + \
    gg.stat_summary(fun_y=base.mean, geom="bar", position="dodge") + \
    gg.stat_summary(fun_data=ggplot2.mean_se, geom="errorbar", position=gg.position_dodge(width=0.90), width=0.2) + \
    gg.labs(x=indep_name, y=dep_name)
    return(plt)
    
def mean_ci(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
    
def t_1samp_table(var_list, var_names, mu=0.0):
    table = pd.DataFrame(index=['mean','SD','Shapiro-Wilk (W)','p(W)','t','p(t)','Wilcoxon (W)','p(W)','ci_low','ci_high'])
    for i, var in enumerate(var_list):
        m = np.mean(var)
        std = np.std(var)
        norm = stats.shapiro(var)
        t_test = stats.ttest_1samp(var, mu)
        wilcox = stats.wilcoxon(var - mu)
        ci = mean_ci(var)
        var_table = pd.DataFrame([m, std, norm[0], norm[1], t_test[0], t_test[1], wilcox[0], wilcox[1], ci[1], ci[2]], index=['mean','SD','Shapiro-Wilk (W)','p(W)','t','p(t)','Wilcoxon (W)','p(W)','ci_low','ci_high'])
        table = pd.concat([table, var_table], axis=1)
    table.columns=var_names
    return table