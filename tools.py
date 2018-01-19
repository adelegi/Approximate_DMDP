import numpy as np
import pickle
import matplotlib.pyplot as plt

from Approximate_DMDP.DMDP_class import create_random_DMDP


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
    return obj

def analyze_time(times, param):
    res = []
    value = list(times.keys())
    res = [np.mean(times[x]) for x in times]
    
    plt.figure()
    plt.plot(value, res)
    plt.title("Execution time when the parameter {} evolves".format(param))
    plt.xlabel(param)
    plt.ylabel("Execution time")
    plt.plot()
    
    time_by_param = [res[i] / value[i] for i in range(len(times))]
    
    plt.figure()
    plt.plot(time_by_param)
    plt.title("Execution time by {}".format(param))
    plt.ylabel("Execution time divide by the nuber of {}".format(param))
    plt.plot()
    
    return res, value

def analyze_m(analys, param):
    m_hist_list = {}

    for x in analys:
        m_hist_list[x] = []
        for i in range(len(analys[x])):
            m_hist = [np.mean(analys[x][i]['m_hist'][k]) for k in range(len(analys[x][i]['m_hist']))]
            m_hist_list[x].append(m_hist)

        m_hist_list[x] = np.array(m_hist_list[x])
        m_hist_list[x] = [np.mean(m_hist_list[x][:, k]) for k in range(len(m_hist_list[x][0]))]
    
    plt.figure()
    for x in m_hist_list:
        plt.plot(m_hist_list[x], label=str(x))
    plt.legend()
    plt.title("Number of iteration during the estimation of p_a(i)*v (ApxTrans), for different value of {}".format(param))
    plt.plot()
    
    return m_hist_list

