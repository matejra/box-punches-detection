from __future__ import division
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
import pywt
import pywt.data
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


np.set_printoptions(threshold=np.nan)
punch_window_size = 24
punch_window_size_half = int(punch_window_size / 2)

################################ Load data ####################################
zuritatrainingl1=np.genfromtxt('data/original/zurita_left__1.csv', delimiter=',')
zuritatrainingr1=np.genfromtxt('data/original/zurita_right__1.csv', delimiter=',')
moytrainingl1=np.genfromtxt('data/original/moy_left__1.csv', delimiter=',')
moytrainingr1=np.genfromtxt('data/original/moy_right__1.csv', delimiter=',')
javitrainingl1=np.genfromtxt('data/original/javi_left__1.csv', delimiter=',')
javitrainingr1=np.genfromtxt('data/original/javi_right__1.csv', delimiter=',')
elenatrainingl1=np.genfromtxt('data/original/elena_left__1.csv', delimiter=',')
elenatrainingr1=np.genfromtxt('data/original/elena_right__1.csv', delimiter=',')
estertrainingl1=np.genfromtxt('data/original/ester_left__1.csv', delimiter=',')
estertrainingr1=np.genfromtxt('data/original/ester_right__1.csv', delimiter=',')
ztestl1=np.genfromtxt('data/original/ztest_left__1.csv', delimiter=',')
ztestr1=np.genfromtxt('data/original/ztest_right__1.csv', delimiter=',')

t_between_samples_l_zurita = zuritatrainingl1[:, 3]
t_between_samples_r_zurita = zuritatrainingr1[:, 3]
t_between_samples_l_moy = moytrainingl1[:, 3]
t_between_samples_r_moy = moytrainingr1[:, 3]
t_between_samples_l_javi = javitrainingl1[:, 3]
t_between_samples_r_javi = javitrainingr1[:, 3]
t_between_samples_l_elena = elenatrainingl1[:, 3]
t_between_samples_r_elena = elenatrainingr1[:, 3]
t_between_samples_l_ester = estertrainingl1[:, 3]
t_between_samples_r_ester = estertrainingr1[:, 3]
t_between_samples_l_ztest = ztestl1[:, 3]
t_between_samples_r_ztest = ztestr1[:, 3]

################################ Absolute time ################################
t_l_zur = np.cumsum(t_between_samples_l_zurita)
t_r_zur = np.cumsum(t_between_samples_r_zurita)
t_l_moy = np.cumsum(t_between_samples_l_moy)
t_r_moy = np.cumsum(t_between_samples_r_moy)
t_l_javi = np.cumsum(t_between_samples_l_javi)
t_r_javi = np.cumsum(t_between_samples_r_javi)
t_l_elena = np.cumsum(t_between_samples_l_elena)
t_r_elena = np.cumsum(t_between_samples_r_elena)
t_l_ester = np.cumsum(t_between_samples_l_ester)
t_r_ester = np.cumsum(t_between_samples_r_ester)
t_l_ztest = np.cumsum(t_between_samples_l_ztest)
t_r_ztest = np.cumsum(t_between_samples_r_ztest)

################ Array of accelerations xyz for l and r arm ###################
ax_l_zurita = zuritatrainingl1[:, 0]
ay_l_zurita = zuritatrainingl1[:, 1]
az_l_zurita = zuritatrainingl1[:, 2]
ax_r_zurita = zuritatrainingr1[:, 0]
ay_r_zurita = zuritatrainingr1[:, 1]
az_r_zurita = zuritatrainingr1[:, 2]
ax_l_moy = moytrainingl1[:, 0]
ay_l_moy = moytrainingl1[:, 1]
az_l_moy = moytrainingl1[:, 2]
ax_r_moy = moytrainingr1[:, 0]
ay_r_moy = moytrainingr1[:, 1]
az_r_moy = moytrainingr1[:, 2]
ax_l_javi = javitrainingl1[:, 0]
ay_l_javi = javitrainingl1[:, 1]
az_l_javi = javitrainingl1[:, 2]
ax_r_javi = javitrainingr1[:, 0]
ay_r_javi = javitrainingr1[:, 1]
az_r_javi = javitrainingr1[:, 2]
ax_l_elena = elenatrainingl1[:, 0]
ay_l_elena = elenatrainingl1[:, 1]
az_l_elena = elenatrainingl1[:, 2]
ax_r_elena = elenatrainingr1[:, 0]
ay_r_elena = elenatrainingr1[:, 1]
az_r_elena = elenatrainingr1[:, 2]
ax_l_ester = estertrainingl1[:, 0]
ay_l_ester = estertrainingl1[:, 1]
az_l_ester = estertrainingl1[:, 2]
ax_r_ester = estertrainingr1[:, 0]
ay_r_ester = estertrainingr1[:, 1]
az_r_ester = estertrainingr1[:, 2]
ax_l_ztest = ztestl1[:, 0]
ay_l_ztest = ztestl1[:, 1]
az_l_ztest = ztestl1[:, 2]
ax_r_ztest = ztestr1[:, 0]
ay_r_ztest = ztestr1[:, 1]
az_r_ztest = ztestr1[:, 2]

########### Re-sampling to uniformly distributed samples ######################
sampling_frequency = 50  # Hz
sampling_time = 1000 / sampling_frequency  # ms

####################### Interpolating data Zurita #############################
n_samples_zurita = round(t_l_zur[-1] / sampling_time)  # End of time divided by 20 (one observation/20ms)
n_samples_zurita = int(n_samples_zurita)
tnew_zurita = np.linspace(20, t_l_zur[-1], n_samples_zurita)
ax_interpolated_r_zurita = signal.resample(ax_r_zurita, n_samples_zurita)
ay_interpolated_r_zurita = signal.resample(ay_r_zurita, n_samples_zurita)
az_interpolated_r_zurita = signal.resample(az_r_zurita, n_samples_zurita)
ax_interpolated_l_zurita = signal.resample(ax_l_zurita, n_samples_zurita)
ay_interpolated_l_zurita = signal.resample(ay_l_zurita, n_samples_zurita)
az_interpolated_l_zurita = signal.resample(az_l_zurita, n_samples_zurita)
interpolated_data_zurita = [ax_interpolated_r_zurita, ay_interpolated_r_zurita, az_interpolated_r_zurita,
                            ax_interpolated_l_zurita, ay_interpolated_l_zurita, az_interpolated_l_zurita]
interpolated_data_zurita = np.array(interpolated_data_zurita)

####################### Interpolating data Moy ################################
n_samples_moy = round(t_l_moy[-1] / sampling_time)  # End of time divided by 20 (one observation/20ms)
n_samples_moy = int(n_samples_moy)
ax_interpolated_r_moy = signal.resample(ax_r_moy, n_samples_moy)
ay_interpolated_r_moy = signal.resample(ay_r_moy, n_samples_moy)
az_interpolated_r_moy = signal.resample(az_r_moy, n_samples_moy)
ax_interpolated_l_moy = signal.resample(ax_l_moy, n_samples_moy)
ay_interpolated_l_moy = signal.resample(ay_l_moy, n_samples_moy)
az_interpolated_l_moy = signal.resample(az_l_moy, n_samples_moy)

interpolated_data_moy = [ax_interpolated_r_moy, ay_interpolated_r_moy, az_interpolated_r_moy,
                            ax_interpolated_l_moy, ay_interpolated_l_moy, az_interpolated_l_moy]
interpolated_data_moy = np.array(interpolated_data_moy)

##################### Interpolating data Javi #################################
n_samples_javi = round(t_l_javi[-1] / sampling_time)  # End of time divided by 20 (one observation/20ms)
n_samples_javi = int(n_samples_javi)
ax_interpolated_r_javi = signal.resample(ax_r_javi, n_samples_javi)
ay_interpolated_r_javi = signal.resample(ay_r_javi, n_samples_javi)
az_interpolated_r_javi = signal.resample(az_r_javi, n_samples_javi)
ax_interpolated_l_javi = signal.resample(ax_l_javi, n_samples_javi)
ay_interpolated_l_javi = signal.resample(ay_l_javi, n_samples_javi)
az_interpolated_l_javi = signal.resample(az_l_javi, n_samples_javi)

interpolated_data_javi = [ax_interpolated_r_javi, ay_interpolated_r_javi, az_interpolated_r_javi,
                            ax_interpolated_l_javi, ay_interpolated_l_javi, az_interpolated_l_javi]
interpolated_data_javi = np.array(interpolated_data_javi)

##################### Interpolating data Elena #################################
n_samples_elena = round(t_l_elena[-1] / sampling_time)  # End of time divided by 20 (one observation/20ms)
n_samples_elena = int(n_samples_elena)
ax_interpolated_r_elena = signal.resample(ax_r_elena, n_samples_elena)
ay_interpolated_r_elena = signal.resample(ay_r_elena, n_samples_elena)
az_interpolated_r_elena = signal.resample(az_r_elena, n_samples_elena)
ax_interpolated_l_elena = signal.resample(ax_l_elena, n_samples_elena)
ay_interpolated_l_elena = signal.resample(ay_l_elena, n_samples_elena)
az_interpolated_l_elena = signal.resample(az_l_elena, n_samples_elena)

interpolated_data_elena = [ax_interpolated_r_elena, ay_interpolated_r_elena, az_interpolated_r_elena,
                            ax_interpolated_l_elena, ay_interpolated_l_elena, az_interpolated_l_elena]
interpolated_data_elena = np.array(interpolated_data_elena)

##################### Interpolating data Ester #################################
n_samples_ester = round(t_l_ester[-1] / sampling_time)  # End of time divided by 20 (one observation/20ms)
n_samples_ester = int(n_samples_ester)
ax_interpolated_r_ester = signal.resample(ax_r_ester, n_samples_ester)
ay_interpolated_r_ester = signal.resample(ay_r_ester, n_samples_ester)
az_interpolated_r_ester = signal.resample(az_r_ester, n_samples_ester)
ax_interpolated_l_ester = signal.resample(ax_l_ester, n_samples_ester)
ay_interpolated_l_ester = signal.resample(ay_l_ester, n_samples_ester)
az_interpolated_l_ester = signal.resample(az_l_ester, n_samples_ester)

interpolated_data_ester = [ax_interpolated_r_ester, ay_interpolated_r_ester, az_interpolated_r_ester,
                            ax_interpolated_l_ester, ay_interpolated_l_ester, az_interpolated_l_ester]
interpolated_data_ester = np.array(interpolated_data_ester)

##################### Interpolating data Zurita test ###########################
n_samples_ztest = round(t_l_ztest[-1] / sampling_time)  # End of time divided by 20 (one observation/20ms)
n_samples_ztest = int(n_samples_ztest)
n_samples_ztest=min(len(ay_r_ztest),len(ay_l_ztest))
ax_interpolated_r_ztest = signal.resample(ax_r_ztest, n_samples_ztest)
ay_interpolated_r_ztest = signal.resample(ay_r_ztest, n_samples_ztest)
az_interpolated_r_ztest = signal.resample(az_r_ztest, n_samples_ztest)
ax_interpolated_l_ztest = signal.resample(ax_l_ztest, n_samples_ztest)
ay_interpolated_l_ztest = signal.resample(ay_l_ztest, n_samples_ztest)
az_interpolated_l_ztest = signal.resample(az_l_ztest, n_samples_ztest)

interpolated_data_ztest = [ax_interpolated_r_ztest, ay_interpolated_r_ztest, az_interpolated_r_ztest,
                            ax_interpolated_l_ztest, ay_interpolated_l_ztest, az_interpolated_l_ztest]
interpolated_data_ztest = np.array(interpolated_data_ztest)

######################### Putting punches to classes ##########################
class1 = interpolated_data_zurita[:, 6747:9900]
class2 = interpolated_data_zurita[:, 10074:13340]
class3 = interpolated_data_zurita[:, 13812:17580]
class4 = interpolated_data_zurita[:, 17747:20965]
class5 = interpolated_data_zurita[:, 22160:26000]
class6 = interpolated_data_zurita[:, 26400:31250]
class1_moy = interpolated_data_moy[:, 5000:9750]
class2_moy = interpolated_data_moy[:, 10410:16040]
class3_moy = interpolated_data_moy[:, 16890:21280]
class4_moy = interpolated_data_moy[:, 22040:27020]
class5_moy = interpolated_data_moy[:, 28030:31240]
class6_moy = interpolated_data_moy[:, 31810:35660]
class1_javi = interpolated_data_javi[:, 7570:11070]
class2_javi = interpolated_data_javi[:, 3416:6660]
class3_javi = interpolated_data_javi[:, 15550:19350]
class4_javi = interpolated_data_javi[:, 11500:14505]
class5_javi = interpolated_data_javi[:, 23930:26860]
class6_javi = interpolated_data_javi[:, 20270:23470]

class1_elena = interpolated_data_elena[:, 2980:4070]
class2_elena = interpolated_data_elena[:, 4460:5580]
class3_elena = interpolated_data_elena[:, 6130:7295]
class4_elena = interpolated_data_elena[:, 7632:8695]
class5_elena = interpolated_data_elena[:, 9430:10630]
class6_elena = interpolated_data_elena[:, 11440:12630]

class1_ester = interpolated_data_ester[:, 3385:4240]
class2_ester = interpolated_data_ester[:, 4360:5140]
class3_ester = interpolated_data_ester[:, 5580:6515]
class4_ester = interpolated_data_ester[:, 7165:8080]
class5_ester = interpolated_data_ester[:, 8980:9850]
class6_ester = interpolated_data_ester[:, 10640:11630]

class1 = np.append(class1, class1_moy, axis=1)
class2 = np.append(class2, class2_moy, axis=1)
class3 = np.append(class3, class3_moy, axis=1)
class4 = np.append(class4, class4_moy, axis=1)
class5 = np.append(class5, class5_moy, axis=1)
class6 = np.append(class6, class6_moy, axis=1)
class1 = np.append(class1, class1_javi, axis=1)
class2 = np.append(class2, class2_javi, axis=1)
class3 = np.append(class3, class3_javi, axis=1)
class4 = np.append(class4, class4_javi, axis=1)
class5 = np.append(class5, class5_javi, axis=1)
class6 = np.append(class6, class6_javi, axis=1)
class1 = np.append(class1, class1_elena, axis=1)
class2 = np.append(class2, class2_elena, axis=1)
class3 = np.append(class3, class3_elena, axis=1)
class4 = np.append(class4, class4_elena, axis=1)
class5 = np.append(class5, class5_elena, axis=1)
class6 = np.append(class6, class6_elena, axis=1)
class1 = np.append(class1, class1_ester, axis=1)
class2 = np.append(class2, class2_ester, axis=1)
class3 = np.append(class3, class3_ester, axis=1)
class4 = np.append(class4, class4_ester, axis=1)
class5 = np.append(class5, class5_ester, axis=1)
class6 = np.append(class6, class6_ester, axis=1)
classes_array = [class1, class2, class3, class4, class5, class6]

############################# Defining functions ############################## 
def plot_coeffs(data, w, title, use_dwt=True):
    #Show dwt or swt coefficients for given data and wavelet.
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []

    if use_dwt:
        for i in range(5):
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)
    else:
        coeffs = pywt.swt(data, w, 5)  # [(cA5, cD5), ..., (cA1, cD1)]
        for a, d in reversed(coeffs):
            ca.append(a)
            cd.append(d)

    fig = plt.figure()
    ax_main = fig.add_subplot(len(ca) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, x in enumerate(ca):
        ax = fig.add_subplot(len(ca) + 1, 2, 3 + i * 2)
        ax.plot(x, 'r.')
        ax.set_ylabel("A%d" % (i + 1))
        if use_dwt:
            ax.set_xlim(0, len(x) - 1)
        else:
            ax.set_xlim(w.dec_len * i, len(x) - 1 - w.dec_len * i)

    for i, x in enumerate(cd):
        ax = fig.add_subplot(len(cd) + 1, 2, 4 + i * 2)
        ax.plot(x, 'g.')
        ax.set_ylabel("D%d" % (i + 1))
        # Scale axes
        ax.set_xlim(0, len(x) - 1)
        if use_dwt:
            ax.set_ylim(min(0, 1.4 * min(x)), max(0, 1.4 * max(x)))
        else:
            vals = x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)]
            ax.set_ylim(min(0, 2 * min(vals)), max(0, 2 * max(vals)))
# Show DWT coefficients
use_dwt = True
#plot_coeffs(data2, 'db5', "DWT: Frequency and phase change - Symmlets5",
#           use_dwt)
#plt.show()

def get_waveletfeatures(data, w, use_dwt=True):
    #Show dwt or swt coefficients for given data and wavelet.
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []

    if use_dwt:
        for i in range(5):
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)
    else:
        coeffs = pywt.swt(data, w, 5)  # [(cA5, cD5), ..., (cA1, cD1)]
        for a, d in reversed(coeffs):
            ca.append(a)
            cd.append(d)
    
    wave_features = []
    for i in range(len(ca)): #ca1 - ca5
        c_Dsquares = []
        for j in range(len(cd[i])):
            c_Dsquares.append((cd[i][j])**2)
        c_Dsumsquares=sum(c_Dsquares)
        wave_features.append(c_Dsumsquares)
    return wave_features # Returns 

def plot_statistical_features(arg1,arg2,arg3,arg4,arg5,arg6,title):
    plt.figure()
    for j in range(6):
        classnum=j+1
        plt.subplot(321)
        plt.plot(np.arange(len(arg1[j])), arg1[j], '.', ms=10, label='class %s' % classnum)
        plt.title('ax right %s' % title)
        plt.subplot(322)
        plt.plot(np.arange(len(arg2[j])), arg2[j], '.', ms=10, label='class %s' % classnum)
        plt.title('ay right %s' % title)
        plt.subplot(323)
        plt.plot(np.arange(len(arg3[j])), arg3[j], '.', ms=10, label='class %s' % classnum)
        plt.title('az right %s' % title)
        plt.subplot(324)
        plt.plot(np.arange(len(arg4[j])), arg4[j], '.', ms=10, label='class %s' % classnum)
        plt.title('ax left %s' % title)
        plt.subplot(325)
        plt.plot(np.arange(len(arg5[j])), arg5[j], '.', ms=10, label='class %s' % classnum)
        plt.title('ay left %s')
        plt.subplot(326)
        plt.plot(np.arange(len(arg6[j])), arg6[j], '.', ms=10, label='class %s' % classnum)
        plt.title('az left %s' % title)
    plt.legend(loc='center', bbox_to_anchor=(1.1, 1.05))
    plt.show()
mode = pywt.Modes.sp1DWT = 1

### Take 2000 minimums from all the classes and define statistical ft. list ###
indexes_2000 = [i for i in range(2000)]
means_list = []
mins_list = []
max_list = []
std_list = []
median_list = []
punches_list = []

threshold_array = [-18, -30, -14, -30, -14, -25]
for i in range(6): # For loop over six classes
    current_class = classes_array[i][:]
    # Take accelerations from the punching arm
    if min(current_class[1, :]) < min(current_class[4, :]):
        indexes = np.argsort(current_class[1, :])
        current_class_ay = current_class[1, :]
    else:
        indexes = np.argsort(current_class[4, :])
        current_class_ay = current_class[4, :]
    list_of_min_indexes = indexes[indexes_2000]
    ind_to_delete = []
    # Deleting indexes (minimums) that are too close to each other (finding local minimum)
    for a, b in itertools.combinations(list_of_min_indexes, 2):
        if a - 5 < b and a + 5 > b:
            a_ind = np.where(list_of_min_indexes == a)
            b_ind = np.where(list_of_min_indexes == b)
            if a_ind < b_ind:
                ind_to_delete.append(b_ind[0][0])
            elif a_ind > b_ind:
                ind_to_delete.append(a_ind[0][0])

    new_list_of_min_indexes = np.delete(list_of_min_indexes, ind_to_delete)
    new_list_of_min_indexes = new_list_of_min_indexes[0:280] #Take first 280 (Max number of pun./class)

    current_class_punches_min_points = current_class[:, new_list_of_min_indexes]
    # Take y-acceleration from punching arm
    if min(current_class[1, :]) < min(current_class[4, :]):
        current_class_punches_min_points_ay = current_class_punches_min_points[1, :]
    else:
        current_class_punches_min_points_ay = current_class_punches_min_points[4, :]
        
    # Thresholding minimum values
    current_class_real_punches_min_points = []
    current_class_real_time = []
    thresholded_new_list_of_min_indexes = []

    for j in range(len(current_class_punches_min_points_ay)):
        if current_class_punches_min_points_ay[j] < threshold_array[i]:  # Punching arm thresholding
            current_class_real_punches_min_points.append(current_class_punches_min_points[:, j])
            thresholded_new_list_of_min_indexes.append(new_list_of_min_indexes[j])

    current_class_real_punches_min_points = np.array(current_class_real_punches_min_points)
    current_class_real_time = np.array(current_class_real_time)
    
    current_class_whole_punch = np.zeros((len(current_class_real_punches_min_points), 6, punch_window_size))
    current_class_whole_punch_mean = np.zeros((len(current_class_real_punches_min_points), 6))
    current_class_whole_punch_minimum = np.zeros((len(current_class_real_punches_min_points), 6))
    current_class_whole_punch_maximum = np.zeros((len(current_class_real_punches_min_points), 6))
    current_class_whole_punch_std = np.zeros((len(current_class_real_punches_min_points), 6))
    current_class_whole_punch_median = np.zeros((len(current_class_real_punches_min_points), 6))
    punch_freq = np.zeros((len(current_class_real_punches_min_points), 6))
    
    for j in range(len(current_class_real_punches_min_points)):
        current_class_whole_punch[j, :, :] = current_class[:, thresholded_new_list_of_min_indexes[j] - punch_window_size_half:
                                                               thresholded_new_list_of_min_indexes[j] + punch_window_size_half]

    for j in range(len(current_class_whole_punch)):
        current_class_whole_punch_mean[j,:]=np.mean(current_class_whole_punch[:][j],axis=1)
        current_class_whole_punch_minimum[j,:]=np.amin(current_class_whole_punch[:][j],axis=1)
        current_class_whole_punch_maximum[j,:]=np.amax(current_class_whole_punch[:][j],axis=1)
        current_class_whole_punch_std[j,:]=np.std(current_class_whole_punch[:][j],axis=1)
        current_class_whole_punch_median[j,:]=np.median(current_class_whole_punch[:][j],axis=1)
    #print(current_class_single_punch_minimum)
    means_list_this_class = np.zeros((len(thresholded_new_list_of_min_indexes), 6))
    mins_list_this_class = np.zeros((len(thresholded_new_list_of_min_indexes), 6))
    max_list_this_class = np.zeros((len(thresholded_new_list_of_min_indexes), 6))
    std_list_this_class = np.zeros((len(thresholded_new_list_of_min_indexes), 6))
    median_list_this_class = np.zeros((len(thresholded_new_list_of_min_indexes), 6))
    
    means_list_this_class =[current_class_whole_punch_mean[:,0],
                       current_class_whole_punch_mean[:,1],
                       current_class_whole_punch_mean[:,2],
                       current_class_whole_punch_mean[:,3],
                       current_class_whole_punch_mean[:,4],
                       current_class_whole_punch_mean[:,5]]
    mins_list_this_class = [current_class_whole_punch_minimum[:,0],
                       current_class_whole_punch_minimum[:,1],
                       current_class_whole_punch_minimum[:,2],
                       current_class_whole_punch_minimum[:,3],
                       current_class_whole_punch_minimum[:,4],
                       current_class_whole_punch_minimum[:,5]]
    max_list_this_class = [current_class_whole_punch_maximum[:,0],
                       current_class_whole_punch_maximum[:,1],
                       current_class_whole_punch_maximum[:,2],
                       current_class_whole_punch_maximum[:,3],
                       current_class_whole_punch_maximum[:,4],
                       current_class_whole_punch_maximum[:,5]]
    
    std_list_this_class = [current_class_whole_punch_std[:,0],
                       current_class_whole_punch_std[:,1],
                       current_class_whole_punch_std[:,2],
                       current_class_whole_punch_std[:,3],
                       current_class_whole_punch_std[:,4],
                       current_class_whole_punch_std[:,5]]
    
    median_list_this_class = [current_class_whole_punch_median[:,0],
                       current_class_whole_punch_median[:,1],
                       current_class_whole_punch_median[:,2],
                       current_class_whole_punch_median[:,3],
                       current_class_whole_punch_median[:,4],
                       current_class_whole_punch_median[:,5]]    
    
    punches_list.append(current_class_whole_punch)
    means_list.append(means_list_this_class)
    max_list.append(max_list_this_class)
    mins_list.append(mins_list_this_class)
    std_list.append(std_list_this_class)
    median_list.append(median_list_this_class)
    
    #plt.plot(current_class_ay, 'b-', current_class_whole_punch_minimum[:,1], 'r.')
    #plt.show()
    #for j in range(len(current_class_real_punches_min_points)):
    #    plt.plot(current_class_whole_punch_ay[j])
    
    #plt.plot(current_class_ay)
    #plt.plot(thresholded_new_list_of_min_indexes,current_class_whole_punch_minimum[:,1],'r.','ms=10')
    
    #plt.show()
# end for
    


"""plot_statistical_features(means_list_ax1,means_list_ay1,means_list_az1,means_list_ax2,means_list_ay2,means_list_az2,'means')
plot_statistical_features(mins_list_ax1,mins_list_ay1,mins_list_az1,mins_list_ax2,mins_list_ay2,mins_list_az2,'mins')
plot_statistical_features(max_list_ax1,max_list_ay1,max_list_az1,max_list_ax2,max_list_ay2,max_list_az2,'max')
"""

########################## Wavelet for training data #########################
# For each punch, get_waveletfeatures for all 6 (3) accelerations
wf=[]
wf_ax1=[]
wf_ay1=[]
wf_az1=[]
wf_ax2=[]
wf_ay2=[]
wf_az2=[]
for i in range(6): #classes
    # Create empty array numpy, append
    for j in range(len(punches_list[i])): #N of punch
        for k in range(6): #Acc. axis
            wf.append(get_waveletfeatures(punches_list[i][j][k], 'db3', use_dwt))
                


for i in range(0,len(wf),6):
    wf_ax1.append(wf[i])
    wf_ay1.append(wf[i+1])
    wf_az1.append(wf[i+2])
    wf_ax2.append(wf[i+3])
    wf_ay2.append(wf[i+4])
    wf_az2.append(wf[i+5])

wf_ax1_c=[]
wf_ay1_c=[]
wf_az1_c=[]
wf_ax2_c=[]
wf_ay2_c=[]
wf_az2_c=[]

# Five plots with 6 subplots (5 wavelet features, each plot with subplot of 6 acc.)
for k in range(5):
    plt.figure()
    startind=0
    stopind=punches_list[0].shape[0]
    wcoef=k+1
    for j in range(6): #classes j, wawelet C[k] access wf_ax1[class&wavelet][punch]
        # wf_a[0][5] -> Fifth punch, class 1, wavelet C1. wf_a[12][5] -> Fifth punch, class1, wave C3
        classnum=j+1
        wf_ax1_c.append([i[k] for i in wf_ax1[startind:stopind]])
        wf_ay1_c.append([i[k] for i in wf_ay1[startind:stopind]])
        wf_az1_c.append([i[k] for i in wf_az1[startind:stopind]])
        wf_ax2_c.append([i[k] for i in wf_ax2[startind:stopind]])
        wf_ay2_c.append([i[k] for i in wf_ay2[startind:stopind]])
        wf_az2_c.append([i[k] for i in wf_az2[startind:stopind]])
        """plt.subplot(321)
        plt.plot(np.arange(punches_list[j].shape[0]), wf_ax1_c[k*6+j], '.', ms=10, label='class %s' % classnum)
        plt.title('ax waveletcoef C%s' % wcoef)
        plt.subplot(322)
        plt.plot(np.arange(punches_list[j].shape[0]), wf_ay1_c[k*6+j], '.', ms=10, label='class %s' % classnum)
        plt.title('ay waveletcoef C%s' % wcoef)
        plt.subplot(323)
        plt.plot(np.arange(punches_list[j].shape[0]), wf_az1_c[k*6+j], '.', ms=10, label='class %s' % classnum)
        plt.title('az waveletcoef C%s' % wcoef)
        plt.subplot(324)
        plt.plot(np.arange(punches_list[j].shape[0]), wf_ax2_c[k*6+j], '.', ms=10, label='class %s' % classnum)
        plt.title('ax2 waveletcoef C%s' % wcoef)
        plt.subplot(325)
        plt.plot(np.arange(punches_list[j].shape[0]), wf_ay2_c[k*6+j], '.', ms=10, label='class %s' % classnum)
        plt.title('ay2 waveletcoef C%s' % wcoef)
        plt.subplot(326)
        plt.plot(np.arange(punches_list[j].shape[0]), wf_az2_c[k*6+j], '.', ms=10, label='class %s' % classnum)
        plt.title('az2 waveletcoef C%s' % wcoef)"""
        if j != 5:
            startind=stopind
            stopind=startind+punches_list[j+1].shape[0]
    """plt.legend(loc='center', bbox_to_anchor=(1.1, 1.05))
    plt.show()"""
    
plt.close("all")

            
nmb_feat=60
n_train_samples=min(len(punches_list[0]),len(punches_list[1]),len(punches_list[2]),len(punches_list[3]),len(punches_list[4]),len(punches_list[5]))
X=np.zeros((n_train_samples*6,nmb_feat))
y=np.zeros((n_train_samples*6))

for i in range(6): #axis
    for j in range(n_train_samples):
        """X[j+n_train_samples*i,:]=[wf_ax1_c[i][j],wf_ay1_c[i][j],wf_az1_c[i][j],wf_ax2_c[i][j],wf_ay2_c[i][j],wf_az2_c[i][j],
         wf_ax1_c[6+i][j],wf_ay1_c[6+i][j],wf_az1_c[6+i][j],wf_ax2_c[6+i][j],wf_ay2_c[6+i][j],wf_az2_c[6+i][j],
         wf_ax1_c[12+i][j],wf_ay1_c[12+i][j],wf_az1_c[12+i][j],wf_ax2_c[12+i][j],wf_ay2_c[12+i][j],wf_az2_c[12+i][j],
         wf_ax1_c[18+i][j],wf_ay1_c[18+i][j],wf_az1_c[18+i][j],wf_ax2_c[18+i][j],wf_ay2_c[18+i][j],wf_az2_c[18+i][j],
         wf_ax1_c[24+i][j],wf_ay1_c[24+i][j],wf_az1_c[24+i][j],wf_ax2_c[24+i][j],wf_ay2_c[24+i][j],wf_az2_c[24+i][j]]
        """
        """X[j+n_train_samples*i,:]=[mins_list[0][i][j],mins_list[1][i][j],mins_list[2][i][j],mins_list[3][i][j], 
         mins_list[4][i][j],mins_list[5][i][j],max_list[0][i][j],max_list[1][i][j],max_list[2][i][j],max_list[3][i][j], 
         max_list[4][i][j],max_list[5][i][j],std_list[0][i][j],std_list[1][i][j],std_list[2][i][j],std_list[3][i][j], 
         std_list[4][i][j],std_list[5][i][j],means_list[0][i][j],means_list[1][i][j],means_list[2][i][j],means_list[3][i][j], 
         means_list[4][i][j],means_list[5][i][j],median_list[0][i][j],median_list[1][i][j],median_list[2][i][j],median_list[3][i][j], 
         median_list[4][i][j],median_list[5][i][j]]"""
        
        X[j+n_train_samples*i,:]=[mins_list[0][i][j],mins_list[1][i][j],mins_list[2][i][j],mins_list[3][i][j], 
         mins_list[4][i][j],mins_list[5][i][j],max_list[0][i][j],max_list[1][i][j],max_list[2][i][j],max_list[3][i][j], 
         max_list[4][i][j],max_list[5][i][j],std_list[0][i][j],std_list[1][i][j],std_list[2][i][j],std_list[3][i][j], 
         std_list[4][i][j],std_list[5][i][j],means_list[0][i][j],means_list[1][i][j],means_list[2][i][j],means_list[3][i][j], 
         means_list[4][i][j],means_list[5][i][j],median_list[0][i][j],median_list[1][i][j],median_list[2][i][j],median_list[3][i][j], 
         median_list[4][i][j],median_list[5][i][j],wf_ax1_c[i][j],wf_ay1_c[i][j],wf_az1_c[i][j],wf_ax2_c[i][j],wf_ay2_c[i][j],wf_az2_c[i][j],
         wf_ax1_c[6+i][j],wf_ay1_c[6+i][j],wf_az1_c[6+i][j],wf_ax2_c[6+i][j],wf_ay2_c[6+i][j],wf_az2_c[6+i][j],
         wf_ax1_c[12+i][j],wf_ay1_c[12+i][j],wf_az1_c[12+i][j],wf_ax2_c[12+i][j],wf_ay2_c[12+i][j],wf_az2_c[12+i][j],
         wf_ax1_c[18+i][j],wf_ay1_c[18+i][j],wf_az1_c[18+i][j],wf_ax2_c[18+i][j],wf_ay2_c[18+i][j],wf_az2_c[18+i][j],
         wf_ax1_c[24+i][j],wf_ay1_c[24+i][j],wf_az1_c[24+i][j],wf_ax2_c[24+i][j],wf_ay2_c[24+i][j],wf_az2_c[24+i][j]]
        y[j+n_train_samples*i]=i
        
        """X[j+n_train_samples*i,:]=[mins_list[0][i][j],mins_list[1][i][j],mins_list[2][i][j],mins_list[3][i][j], 
         mins_list[4][i][j],mins_list[5][i][j],max_list[0][i][j],max_list[1][i][j],max_list[2][i][j],max_list[3][i][j], 
         max_list[4][i][j],max_list[5][i][j]]"""
        y[j+n_train_samples*i]=i
        

###################### Comparison of different classifiers ####################
X_scaled = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)
C = 1.0 # SVM regularization parameter
svm_model = svm.SVC(kernel='linear', C=1, cache_size=1000)
scores = cross_val_score(svm_model, X_scaled, y, cv=10)
print("Accuracy SVM cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
svm_model.fit(X_train, y_train) 
print("Accuracy SVM train-test 70-30: %0.2f" % (svm_model.score(X_test, y_test)))

neigh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(neigh, X_scaled, y, cv=10)
print("Accuracy K nearest cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
neigh.fit(X_train, y_train) 
print("Accuracy K nearest train-test 70-30: %0.2f" % (neigh.score(X_test, y_test)))

clf3 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=20, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
clf3.fit(X_train, y_train)
print("Accuracy Random forest train-test 70-30: %0.2f" % clf3.score(X_test, y_test))
scores = cross_val_score(clf3, X_scaled, y, cv=10)
print("Accuracy Random forest: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = neigh.fit(X_train, y_train).predict(X_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,2,3,4,5,6],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,2,3,4,5,6], normalize=True,
                      title='Normalized confusion matrix')

plt.show()