import os
import time
import datetime
import numpy as np

import pandas as pd
from xlrd.biffh import XLRDError
import warnings

warnings.filterwarnings("ignore")


# 设置根目录路径
root_path = os.path.abspath('.')
# print('The Root Path:', root_path)

# 数据路径
raw_data_dir_path = os.path.join(root_path, "data/rawdata")

# label 设置
raw_data_col_name = "Time,A: EKG,TT-AV Sync - 1B,C: EEG,D: EMG,E: Skin Cond,F: Temp,G: Abd Resp," \
                    "MyoScan-Pro 400 - 1H,A: EKG HR (Smoothed),A: EKG VLF Total power,A: EKG LF Total power," \
                    "A: EKG HF Total power,D: EMG + smoothing,A: EKG VLF total power mean," \
                    "A: EKG LF total power mean,A: EKG HF Total power mean,A: EKG LF/HF (means)," \
                    "D: EMG mean (uV),E: Skin conductance mean (uS),E: SC as % of value mean (%),F: " \
                    "Temperature mean (Deg)".split(",")


def create_state_label(n=40):
    empty_label = ['E_0']
    state_label = empty_label.copy()
    # arousal_label = ['F_A', 'A_0']
    # valence_label = ['A_V', 'V_0']
    arousal_label = ['A_0']
    valence_label = ['V_0']
    for idx in range(1, n + 1):
        empty_label.append('E_' + str(idx))
        state_label.extend(['F_' + str(idx), 'E_' + str(idx)])
        arousal_label.append("A_" + str(idx))
        valence_label.append("V_" + str(idx))

    # empty_label += ['F_A', 'A_0'] + ['A_V', 'V_0']
    empty_label += ['A_0'] + ['V_0']
    state_label += arousal_label + valence_label
    return state_label, empty_label


def read_time_marker():
    time_marker_path = os.path.join(root_path, "data/time.txt")
    data = pd.read_table(time_marker_path, names=['index', 'name', 'F', 'A', 'V'],
                         index_col=0, header=None, encoding='gb2312', delimiter=",")
    return data


def cut_rawdata(f_name, s_label, fav_start_no=88, pre_point=16, end_point=80):

    # 导入数据
    raw_data_temp_path = os.path.join(raw_data_dir_path, f_name)

    # 不导入列名，而是使用前面设置的 label
    raw_data = pd.read_table(raw_data_temp_path, header=None, delimiter=",")
    raw_data.columns = raw_data_col_name

    # TT-AV 关于正负的判断，分为两类
    raw_data_state = np.array(raw_data["TT-AV Sync - 1B"]) < 0
    raw_data["TT-AV-TF"] = raw_data_state

    print(raw_data_state, len(raw_data["TT-AV Sync - 1B"]))
    print(np.diff(raw_data_state), sum(np.diff(raw_data_state) != 0))
    print(len(np.append(0, np.cumsum(np.diff(raw_data_state)))))
    # print(raw_data[2].value_counts())

    # 对于 TT-AV-TF 这列，我们进行差分后累和并在最前面补 0（确保长度相等），得到分段的标号
    raw_data_state_cut = np.append(0, np.cumsum(np.diff(raw_data_state)))
    raw_data["TT-AV-TF-No"] = raw_data_state_cut

    # 对应三个实验相关位点的切分
    fig_split = np.logical_and(raw_data_state_cut >= 88 - 1, raw_data_state_cut <= 88 + 80 - 1)
    aro_split = np.logical_and(raw_data_state_cut >= 171 - 1, raw_data_state_cut <= 171 + 40 - 1)
    val_split = np.logical_and(raw_data_state_cut >= 213 - 1, raw_data_state_cut <= 213 + 40 - 1)
    all_split = fig_split + aro_split + val_split

    # 标号的生成
    fig_cut = raw_data_state_cut[fig_split]
    aro_cut = raw_data_state_cut[aro_split]
    val_cut = raw_data_state_cut[val_split]

    # 各自的差分，2、3 两个实验进行基数补偿
    fig_cut_diff = np.append(0, np.cumsum(np.diff(fig_cut)))
    aro_cut_diff = np.append(0, np.cumsum(np.diff(aro_cut))) + len(set(fig_cut_diff))
    val_cut_diff = np.append(0, np.cumsum(np.diff(val_cut))) + len(set(fig_cut_diff)) + len(set(aro_cut_diff))
    print(fig_cut_diff, len(fig_cut_diff))
    print(aro_cut_diff, len(aro_cut_diff))
    print(val_cut_diff, len(val_cut_diff))

    # 合并实验状态的差分集
    all_cut_diff = np.concatenate((fig_cut_diff, aro_cut_diff, val_cut_diff), axis=0)
    state_name = [s_label[id] for id in all_cut_diff]
    raw_data["State_Name"] = "None"     # 不涉及的先打一个其余标签
    raw_data["State_Name"].iloc[all_split] = state_name
    # raw_data["State_Name"] = state_name

    # 反推索引列
    cut_index = raw_data.iloc[
        np.logical_and(raw_data_state_cut >= 88, raw_data_state_cut <= 213 + 40 - 2)
    ].index.tolist()
    pre_list = list(range(min(cut_index) - pre_point, min(cut_index)))  # 前推 pre_point 个点
    end_list = list(range(max(cut_index) + 1, max(cut_index) + end_point + 1))  # 最后只取 end_point 个点
    cut_index = pre_list + cut_index + end_list

    cut_data = raw_data.iloc[cut_index, :]
    print(cut_data)

    state_counts = cut_data['State_Name'].value_counts()

    cut_data.to_csv(f_name.split(".")[0] + ".csv", index=None)

    return state_counts


def main():
    # 查看 label 的分割以及 label 的数量
    print("Col Label Name:\n", raw_data_col_name,
          "\nTotal Col Labels Number:", len(raw_data_col_name))

    state_label, empty_label = create_state_label(40)
    # print(empty_label)
    print("State Label Name:\n", state_label,
          "\nTotal State Labels Number:", len(state_label))

    data = read_time_marker()
    print(data.loc[1].to_dict().items())

    # file_name = "chenshiyun.txt"
    # state_counts = cut_rawdata(file_name, state_label)
    #
    # for key, values in state_counts.items():
    #     print(key, values)


if __name__ == '__main__':
    main()
