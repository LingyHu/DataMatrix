import os.path
import tkinter as tk
from tkinter import filedialog
import time
import psutil

import cv2
import numpy as np
import pandas as pd
import pylibdmtx.pylibdmtx as dmtx
import statsmodels.api as sm
from sklearn.neighbors import LocalOutlierFactor


# 定义寻峰函数，寻找可能的分割线
def find_peak(y_input):
    # 去基线
    y_sub = np.where(y_input < 2400, 0, y_input)

    # 对y进行差分
    y_diff = np.diff(y_sub)

    # 对y_diff求符号
    y_diff_sign = np.sign(y_diff)

    # 对y_diff_sign再求一次差分
    y_diff_sign_diff = np.diff(y_diff_sign)

    # 遍历y_diff_sign_diff，
    # 将其中y_diff_sign_diff[i]=-2(单峰)或y_diff_sign_diff[i] == -1（梯形峰，有多个相等的最大值）的i值变为i+1后存储到数组peak_list_origin中
    peak_list_origin = []
    for i in range(len(y_diff_sign_diff)):
        if y_diff_sign_diff[i] == -2 or y_diff_sign_diff[i] == -1:
            # 记录位置
            peak_list_origin.append(i + 1)

    # 对peak_list_origin做差分，查看两个峰值之间的间隔，保存在数组peak_list_diff中
    peak_list_diff_origin = np.diff(peak_list_origin)

    # 由于初步的峰存在很多噪声，所以会有很多很小的间隔值。
    # 对peak_list_diff排序，则排在前面很小的值为噪声的可能性很大，越往后越有可能是正确的峰。
    # 因此为了预测出真实的峰间距，需要更多的参考后面的数据，
    # 即对排序后的peak_list_diff的数据，越往后可信度越高。
    # 因此采用指数加权移动平均模型（EWMA）来预测真实的峰间距

    # 对peak_list_diff_origin排序
    peak_list_diff_origin.sort()

    # 将peak_list_diff_origin数据转换为pandas序列
    peak_list_diff = pd.Series(peak_list_diff_origin)

    # 计算指数加权移动平均
    peak_list_diff_ewma = peak_list_diff.ewm(alpha=0.355).mean()

    # 获取平滑后的最后一个值作为预测的峰间距。
    prediction_peak_spacing = peak_list_diff_ewma.iloc[-1]

    # 原始的峰列表由于噪音问题会存在很多肩峰和裂峰的现象。因此我们需要判断哪些峰是同一组的。
    # 将肩峰裂峰放在一个数组内，再从这个数组内找出一个峰来代表真实的峰。
    # 创建一个数组peak_list_group用来存放分好组以后的峰，
    # 创建一个数组temp_list用来分组时存放同一组的峰
    peak_list_find_peak = peak_list_origin.copy()
    peak_list_group = []
    temp_list = [peak_list_find_peak[0]]

    # 遍历peak_list_find_peak每一对相邻的峰，计算它们的差值（即间隔）
    for i in range(len(peak_list_find_peak) - 1):

        # 差分，得到峰间距，根据峰间距来判断是否是同一组的峰
        x_diff = peak_list_find_peak[i + 1] - peak_list_find_peak[i]

        # 因为单个二维码水平或者竖直方向最多有26个模块，图像在256*256的空间内，所以正常情况下的最小间距是256/26大约10，
        # 再结合我们预测的真实峰间距，若峰间距小于max（3，prediction_peak_spacing / 2），则认为是同一组的峰，
        # 直到峰间距大于max（3，prediction_peak_spacing / 2）
        if x_diff <= max(3, prediction_peak_spacing / 2):
            temp_list.append(peak_list_find_peak[i + 1])
        else:
            peak_list_group.append(temp_list)
            temp_list = [peak_list_find_peak[i + 1]]

    # 如果是最后一组峰（肩峰或单峰），则直接将其添加到peak_list_group中，避免漏掉尾峰
    if peak_list_find_peak[-1] - temp_list[-1] == 0:
        peak_list_group.append(temp_list)

    # 找出每组峰里最有代表性的峰作为该组峰真正的峰存储到peak_list_group_max，
    # 这里采取选择强度最大的峰作为真正的峰。
    peak_list_group_max = []

    # 遍历peak_list_group
    for i in range(len(peak_list_group)):

        # 如果只有一个峰，则直接将其添加到peak_list_group_max中
        if len(peak_list_group[i]) == 1:
            peak_list_group_max.append(int(peak_list_group[i][0]))

        # 如果有多个峰，则找出最大值所对应的j值
        else:
            # 如果组内有多个峰，则记录每个峰的y值，选择最大的y值作为代表。
            peak_y_list = []
            for j in range(len(peak_list_group[i])):
                peak_y_list.append(y_sub[int(peak_list_group[i][j])])
            max_y_sub = max(peak_y_list)

            # 如果只有1个max_y_sub，则取最大值所对应的x值
            if peak_y_list.count(max_y_sub) == 1:
                max_y_sub_idx = peak_y_list.index(max_y_sub)
                peak_list_group_max.append(int(peak_list_group[i][max_y_sub_idx]))

            # 如果出现了多个峰的y_sub值相同的情况，则取最大y_sub对应的x值
            else:
                max_y_sub_idx = [i for i, x in enumerate(peak_y_list) if x == max_y_sub]
                peak_list_group_max.append(int(peak_list_group[i][max_y_sub_idx[0]]))

    # 遍历peak_list_group_max,
    # 进行补峰（如果相邻的峰间距大于预期值的两倍，则在它们之间插入新的峰值，直到峰值间距符合预期。）
    # 补峰后，peak_list_group_max长度改变，要注意更新peak_list_group_max长度
    peak_list_len = len(peak_list_group_max)
    i = 0
    while i < peak_list_len - 1:

        # 计算峰间距
        peak_diff = peak_list_group_max[i + 1] - peak_list_group_max[i]

        # 插入峰的个数统计值
        insert_peak_num = 1

        # 如果峰间距大于2倍预测峰间距，则需要插入新的峰
        while peak_diff > 1.4 * prediction_peak_spacing:
            # 插入值.由数据特征可知，由于存在大量噪音，导致预测的峰间距实际上会偏小，因此在插入峰间距时给予一定的权重进行补偿
            insert_value = int(peak_list_group_max[i] + prediction_peak_spacing * 1.2 * insert_peak_num)

            # 在第一个峰间隔预测峰间距处插入一个峰
            peak_list_group_max.insert(i + insert_peak_num, insert_value)

            # 重新计算峰间距
            peak_diff = peak_list_group_max[i + 1 + insert_peak_num] - peak_list_group_max[i + insert_peak_num]

            # 插入峰的个数统计值加1
            insert_peak_num += 1

            # 更新列表长度
            peak_list_len += 1

        i += 1

    # 返回修正后的峰值列表
    return peak_list_group_max


# 定义网格划分函数，得到各网格的四顶点坐标
def grid_division(peak_list_x, peak_list_y, is_point=False):
    # 存储列信息，即每列的左右端点坐标
    column_list = []

    # 遍历peak_list_x
    for i in range(len(peak_list_x) - 1):
        # 通过遍历peak_list_x，每次取出相邻的两个峰值peak_list_x[i]和peak_list_x[i + 1]，
        # 表示一列的左右端点
        temp_list = [peak_list_x[i], peak_list_x[i + 1]]

        # 将暂存的每列的左右端点坐标添加到column_list中
        column_list.append(temp_list)

    # 存储行信息，即每行的上下端点坐标
    row_list = []

    # 遍历peak_list_y
    for i in range(len(peak_list_y) - 1):
        # 暂存每行的上下端点坐标
        temp_list = [peak_list_y[i], peak_list_y[i + 1]]

        # 将暂存的每行的上下端点坐标添加到row_list中
        row_list.append(temp_list)

    # 如果is_point为True，则说明是点状二维码
    # 点状二维码只取奇数行和奇数列，删除column_list和row_list的偶数行和偶数列
    if is_point:
        # 删除column_list的偶数列, 即索引为奇数的列
        column_list = column_list[::2]

        # 删除row_list的偶数行, 即索引为奇数的行
        row_list = row_list[::2]


    grid_points = []

    # 遍历行
    for i in range(len(row_list)):
        # 遍历列
        for j in range(len(column_list)):
            # 计算每个网格的四个顶点坐标
            # 左上角顶点坐标
            left_top = (column_list[j][0], row_list[i][0])

            # 右上角顶点坐标
            right_top = (column_list[j][1], row_list[i][0])

            # 左下角顶点坐标
            left_bottom = (column_list[j][0], row_list[i][1])

            # 右下角顶点坐标
            right_bottom = (column_list[j][1], row_list[i][1])

            # 将每个网格的四个顶点坐标存储到grid_points中
            grid_points.append([left_top, right_top, left_bottom, right_bottom])

    # 返回网格划分后各网格的四顶点坐标
    return grid_points


# 定义网格迭代分割函数，返回分割结果——一个存储着每一层单位面积灰度值的列表
# 若传入的是网格列表，则返回一组单位面积灰度值的列表
def grid_iterative_segmentation(grid_points_list_input, binary_image_input):
    # 定义存储一组单位面积灰度值列表的数组
    gray_value_list_array = []

    # 遍历网格列表，对每一个网格进行迭代分割
    for grid_points in grid_points_list_input:
        # 定义存储每一层单位化灰度值的列表
        gray_value_list = []

        # 获取网格的四个顶点坐标，可以获得该网格的长和宽
        # grid_points[1][0] 表示右上角顶点（right_top）的横坐标（x2）
        grid_length = grid_points[1][0] - grid_points[0][0]
        # grid_points[2][1] 表示左下角顶点（left_bottom）的纵坐标（y2）。
        grid_width = grid_points[2][1] - grid_points[0][1]

        # 迭代分割网格，直到网格的长grid_length<=2或者宽grid_width<=2
        while grid_length > 2 and grid_width > 2:
            # 首先计算当前四个顶点连成的四边形的单位面积灰度值
            # 即累加四个顶点连成的四边形内每个像素点的灰度值，然后除以四边形面积
            # 然后将该单位化灰度值存储到gray_value_list中
            mask = np.zeros_like(binary_image_input)
            cv2.fillPoly(mask, [np.array([grid_points])], 255)
            gray_value = np.mean(binary_image_input[mask > 0])
            gray_value_list.append(gray_value)

            # 再然后将网格的长和宽都-2，得到新的网格，计算新的网格的单位面积灰度值，添加到gray_value_list中
            # 重复上述过程，直到网格的长grid_length<=2或者宽grid_width<=2，
            # 返回gray_value_list

            # 更新网格的长和宽
            # 如果网格的长和宽都大于2，则长和宽都-2
            if grid_length > 2 and grid_width > 2:
                grid_length -= 2
                grid_width -= 2
            # 如果网格得长小于等于2，但是宽大于2，则宽-2
            elif grid_length <= 2 < grid_width:
                grid_width -= 2
            # 如果网格得宽小于等于2，但是长大于2，则长-2
            else:
                grid_length -= 2

            # 重新计算网格的四个顶点坐标
            grid_points = [(grid_points[0][0] + 1, grid_points[0][1] + 1),
                           (grid_points[1][0] - 1, grid_points[1][1] + 1),
                           (grid_points[2][0] + 1, grid_points[2][1] - 1),
                           (grid_points[3][0] - 1, grid_points[3][1] - 1)]

        # 将gray_value_list添加到gray_value_list_array中
        gray_value_list_array.append(gray_value_list)

    # 返回gray_value_list_array
    return gray_value_list_array


# 定义一个预测灰度值的函数predict_gray_value
def predict_gray_value(grey_tend_list_total):
    # 存储根据灰度分布得到的预测灰度值
    predict_grey_list = []

    # 循环遍历每个灰度分布
    for gray_tend_list in grey_tend_list_total:
        # 将灰度分布转换为数组
        # x数组用于表示每个灰度值的位置，y为当前灰度趋势。
        x = np.arange(len(gray_tend_list))
        y = np.array(gray_tend_list)

        # 通过异常点检测算法lof，检测异常点
        # outlier_scores表示每个灰度值的异常程度，outlier_indices则是检测到的异常点的索引。
        clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
        outlier_scores = clf.fit_predict(y.reshape(-1, 1))
        outlier_indices = np.where(outlier_scores == -1)[0]

        # 修正异常点，平滑异常点周围的灰度值。
        smoothed_y = np.copy(y)
        for outlier_idx in outlier_indices:
            if outlier_idx > 0:
                if outlier_idx <= 2:
                    if outlier_idx + 2 < len(y):  # 添加检查
                        smoothed_y[outlier_idx] = 2 * y[outlier_idx + 1] - y[outlier_idx + 2]
                else:
                    if outlier_idx - 2 >= 0:  # 添加检查
                        smoothed_y[outlier_idx] = 2 * y[outlier_idx - 1] - y[outlier_idx - 2]

        # 进行局部加权回归LWR预测，得到二值化后的预测值predict_gray_value_lwr
        model = sm.nonparametric.KernelReg(endog=smoothed_y, exog=x, var_type='o')
        y_fit, _ = model.fit(x)

        # 计算预测点
        x_pred = np.arange(len(gray_tend_list), len(gray_tend_list) + 1)
        y_pred, _ = model.fit(x_pred)
        predict_gray_value_lwr = y_pred[0]

        # 根据阈值调整预测值,将predict_gray_value_lwr二值化，1表示黑色，0表示白色
        if predict_gray_value_lwr < 128:
            predict_gray_value_lwr = 255
        else:
            predict_gray_value_lwr = 0

        # 将预测值添加到predict_grey_list中
        predict_grey_list.append(predict_gray_value_lwr)

    return predict_grey_list


# 定义解码函数，输入图片的路径，输出解码后的数据矩阵
def decode_dmtx(image_path):
    # 记录开始时间
    start_time = time.time()

    # 获取当前进程的内存使用情况
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024  # 转换为KB

    # 采用cv2读取路径为image_path的PNG图片
    image = cv2.imread(image_path)

    # 放缩到256*256的像素空间中
    image = cv2.resize(image, (256, 256))

    # CLAHE（对比度限制的自适应直方图均衡）对比度增强，并将图像转换为灰度图像。
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5, 5))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = clahe.apply(image)

    # 采用cv2的双边滤波方法，去噪,尽可能保留边缘信息
    # 初始化带宽k=5，带宽越小，边缘信息保存得越好
    # denoised_bw控制滤波的带宽。
    denoised_bw = 5
    denoised_image = cv2.bilateralFilter(image, denoised_bw, 75, 75)

    # 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算，将图像转换为黑白图像。
    ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反转二值化图像的颜色，使黑色部分变白，白色部分变黑
    binary_image = cv2.bitwise_not(binary_image)

    # 计算每个像素点与右侧一个像素点的灰度差值绝对值，并统计每一列像素的差值和
    height, width = binary_image.shape[:2] # 获取彩色图片的高宽
    column_sums = np.zeros(width)  # 存储每一列像素的差值和
    row_sums = np.zeros(height)  # 存储每一行像素的差值和
    for y in range(height):
        for x in range(width - 1):
            diff = abs(int(binary_image[y, x]) - int(binary_image[y, x + 1]))  # 计算灰度差值绝对值
            column_sums[x] += diff  # 累加差值到相应列

    # 计算最右侧像素点与0的差值绝对值，并添加到column_sums的最后一列
    column_sums[-1] = abs(int(binary_image[0, -1]) - 0)

    for y in range(height - 1):
        for x in range(width):
            diff = abs(int(binary_image[y, x]) - int(binary_image[y + 1, x]))  # 计算灰度差值绝对值
            row_sums[y] += diff  # 累加差值到相应行

    # 计算最下方像素点与0的差值绝对值，并添加到row_sums的最后一行
    row_sums[-1] = abs(int(binary_image[-1, 0]) - 0)

    # 寻找峰值（找出水平方向和垂直方向的分割线）
    find_peak_list_x = find_peak(column_sums)
    find_peak_list_y = find_peak(row_sums)

    # 由DataMatrix二维码的码制可知，二维码的大小都是偶数个模块，因此分割线的数量应该为奇数。
    # 比较水平方向和竖直方向的分割线数量，如果不相等，则图中噪声较大，需要增大滤波带宽，直到两者相等或者带宽大于30
    # 带宽若是大于30，边缘信息会因为滤波而丢失，无法进行解码
    while len(find_peak_list_x) != len(find_peak_list_y) or len(find_peak_list_x) % 2 == 0 or len(find_peak_list_y) % 2 == 0 and denoised_bw < 30:
        # 增加带宽
        denoised_bw += 2
        denoised_image = cv2.bilateralFilter(image, denoised_bw, 75, 75)

        # 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
        ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算每个像素点与右侧一个像素点的灰度差值绝对值，并统计每一列像素的差值和
        height, width = binary_image.shape[:2]
        column_sums = np.zeros(width)  # 存储每一列像素的差值和
        row_sums = np.zeros(height)  # 存储每一行像素的差值和
        for y in range(height):
            for x in range(width - 1):
                diff = abs(int(binary_image[y, x]) - int(binary_image[y, x + 1]))  # 计算灰度差值绝对值
                column_sums[x] += diff  # 累加差值到相应列

        # 计算最右侧像素点与0的差值绝对值，并添加到column_sums的最后一列
        column_sums[-1] = abs(int(binary_image[0, -1]) - 0)

        for y in range(height - 1):
            for x in range(width):
                diff = abs(int(binary_image[y, x]) - int(binary_image[y + 1, x]))  # 计算灰度差值绝对值
                row_sums[y] += diff  # 累加差值到相应行

        # 计算最下方像素点与0的差值绝对值，并添加到row_sums的最后一行
        row_sums[-1] = abs(int(binary_image[-1, 0]) - 0)

        find_peak_list_x = find_peak(column_sums)
        find_peak_list_y = find_peak(row_sums)

    if len(find_peak_list_x) != len(find_peak_list_y):
        print("二维码噪声过大，无法进行解码")
        return None

    # 得到网格划分后每个网格的四个顶点坐标
    grid_points_list = grid_division(peak_list_x=find_peak_list_x, peak_list_y=find_peak_list_y, is_point=False)

    # 得到网格列表内每个网格的灰度分布趋势
    grid_grey_tend_list = grid_iterative_segmentation(grid_points_list_input=grid_points_list,
                                                      binary_image_input=binary_image)
    # 得到每个网格二值化后的值，1表示黑色，0表示白色
    predict_grey_value_list = predict_gray_value(grid_grey_tend_list)

    # 将预测值转换为二维数组。其中，二维数组的大小是predict_grey_value_list的平方根，
    # 因为二维码是一个n阶矩阵
    matrix_ranks = int(np.sqrt(len(predict_grey_value_list)))
    dmtx_matrix = np.array(predict_grey_value_list).reshape(matrix_ranks, matrix_ranks)

    # 修改第一行的奇数位为0，偶数位为255
    dmtx_matrix[0, 1::2] = 255
    dmtx_matrix[0, 0::2] = 0

    # 修改最后一列的奇数位为255，偶数位为0
    dmtx_matrix[1::2, -1] = 0
    dmtx_matrix[0::2, -1] = 255

    # 修改矩阵的第一列和最后一行
    dmtx_matrix[:, 0] = 0
    dmtx_matrix[-1, :] = 0

    # 记录结束时间
    end_time = time.time()

    # 获取结束时的内存消耗
    final_memory = process.memory_info().rss / 1024

    # 计算执行时间和内存消耗
    execution_time = end_time - start_time
    memory_consumption = final_memory - initial_memory

    # 输出执行时间和内存消耗
    # print(f"Execution Time: {execution_time:.4f} seconds")
    # print(f"Memory Consumption: {memory_consumption:.4f} KB")

    # 返回解码后的数据矩阵+解码时间+内存消耗
    return dmtx_matrix, execution_time, memory_consumption


# root = tk.Tk()
# root.withdraw()
# f_path = filedialog.askopenfilename()

# test_data = decode_dmtx(f_path)

# # 设置灰度图的大小
# length = int(test_data.size ** 0.5)

# # 将灰度值列表（解码数据）转成灰度图像
# predict_grey_value_array = np.array(test_data, dtype=np.uint8)
# # 将数组重塑为一个length x length的二维数组，表示二维码的灰度图像。
# predict_grey_image = predict_grey_value_array.reshape(length, length)

# # 缩放灰度图
# # 将灰度图像的尺寸调整为400x400像素，使用INTER_AREA插值方法进行缩放（适合图像缩小）。
# resized_grey_image = cv2.resize(predict_grey_image, (400, 400), interpolation=cv2.INTER_AREA)

# # 在图像四周添加50像素的空白
# # BORDER_CONSTANT表示使用常数值填充边框，value=0表示边框填充为黑色（0表示黑色）。
# grey_image_with_border = cv2.copyMakeBorder(resized_grey_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

# # 保存路径
# save_path_dir = os.path.join(os.path.dirname(f_path), 'output2')
# os.makedirs(save_path_dir, exist_ok=True)
# # os.path.basename(f_path)获取原文件的文件名
# save_path = os.path.join(save_path_dir, os.path.basename(f_path))

# # 将添加了边框的图像保存到save_path路径。
# cv2.imwrite(save_path, grey_image_with_border)

# # 解码处理后的灰度图
# # 使用pylibdmtx库解码图像，尝试从图像中提取二维码数据，并将解码结果存储在decoded_data中。
# decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))

# # 如果解码失败，将图片边缘进行反色处理后再次解码
# # 解码失败：说明二维码图像可能因为对比度或其他问题未被正确解码。
# if not decoded_data:
#     # 在图像四周添加50像素的空白，这会产生反色效果。
#     grey_image_with_border = cv2.copyMakeBorder(resized_grey_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

#     # 保存反色处理后的图像，并再次尝试解码。
#     save_path_dir = os.path.join(os.path.dirname(f_path), 'output2')
#     os.makedirs(save_path_dir, exist_ok=True)
#     save_path = os.path.join(save_path_dir, os.path.basename(f_path))

#     # 保存图片
#     cv2.imwrite(save_path, grey_image_with_border)

#     # 解码灰度图
#     decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))

# # 打印解码结果
# print(decoded_data)


results = []

root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory(title="选择待处理文件夹")

# 获取文件夹中所有的图像文件
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):  # 检查文件扩展名
        f_path = os.path.join(folder_path, filename)

        try:
            # 解码该图像
            test_data, exec_time, mem_usage = decode_dmtx(f_path)

            # 如果解码成功，处理解码结果
            if test_data is not None and len(test_data) > 0:
                print(f"Decoded data for {filename}: {test_data}")
            else:
                print(f"Failed to decode {filename}")

            if test_data is not None and len(test_data) > 0:
                # 设置灰度图的大小
                # 计算解码数据的边长，假设二维码是一个正方形矩阵，因此计算其边长为test_data.size的平方根。
                length = int(test_data.size ** 0.5)

                # 将灰度值列表（解码数据）转成灰度图像
                predict_grey_value_array = np.array(test_data, dtype=np.uint8)
                # 将数组重塑为一个length x length的二维数组，表示二维码的灰度图像。
                predict_grey_image = predict_grey_value_array.reshape(length, length)

                # 缩放灰度图
                # 将灰度图像的尺寸调整为400x400像素，使用INTER_AREA插值方法进行缩放（适合图像缩小）。
                resized_grey_image = cv2.resize(predict_grey_image, (400, 400), interpolation=cv2.INTER_AREA)

                # 在图像四周添加50像素的空白
                # BORDER_CONSTANT表示使用常数值填充边框，value=0表示边框填充为黑色（0表示黑色）。
                grey_image_with_border = cv2.copyMakeBorder(resized_grey_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

                # 保存路径
                save_path_dir = os.path.join(os.path.dirname(f_path), 'output_timeandspace')
                os.makedirs(save_path_dir, exist_ok=True)
                # os.path.basename(f_path)获取原文件的文件名
                save_path = os.path.join(save_path_dir, os.path.basename(f_path))

                # 将添加了边框的图像保存到save_path路径。
                cv2.imwrite(save_path, grey_image_with_border)

                # 解码处理后的灰度图
                # 使用pylibdmtx库解码图像，尝试从图像中提取二维码数据，并将解码结果存储在decoded_data中。
                decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))

                # 如果解码失败，将图片边缘进行反色处理后再次解码
                # 解码失败：说明二维码图像可能因为对比度或其他问题未被正确解码。
                if not decoded_data:
                    # 在图像四周添加50像素的空白，这会产生反色效果。
                    grey_image_with_border = cv2.copyMakeBorder(resized_grey_image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)

                    # 保存反色处理后的图像，并再次尝试解码。
                    save_path_dir = os.path.join(os.path.dirname(f_path), 'output_timeandspace')
                    os.makedirs(save_path_dir, exist_ok=True)
                    save_path = os.path.join(save_path_dir, os.path.basename(f_path))

                    # 保存图片
                    cv2.imwrite(save_path, grey_image_with_border)

                    # 解码灰度图
                    decoded_data = dmtx.decode(cv2.imread(save_path, cv2.IMREAD_GRAYSCALE))

                # 打印解码结果
                if decoded_data:
                    print(f"Decoded data for {filename}: {test_data}")
                    print(f"Execution Time: {exec_time:.4f} seconds")
                    print(f"Memory Usage: {mem_usage:.4f} MB")
                else:
                    print(f"Failed to decode {filename}")

                results.append([filename, 'Success', exec_time, mem_usage])
        
        except Exception as e:
            # 捕获异常并继续处理下一个文件
            error_message = f"Error processing {filename}: {str(e)}"
            print(error_message)
            results.append([filename, 'Error', error_message])

# 将结果写入Excel文件
df = pd.DataFrame(results, columns=['Filename', 'Status', 'Execution Time (seconds)', 'Memory Usage (KB)'])
output_excel_path = os.path.join(folder_path, 'decoded_results.xlsx')
df.to_excel(output_excel_path, index=False)

print(f"Results saved to {output_excel_path}")