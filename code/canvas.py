import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os.path


# 定义寻峰函数
def find_peak(y_input):
    # 去基线，y＜4000的认为是噪音，扣除
    # 4000的取值原因：最大尺寸26*26，所以1格灰度值400/26*255
    # 至少有1格模式边，就算数据区某行或者某列全为0，所以至少4000
    y_sub = np.where(y_input < 7000, 0, y_input)

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
            peak_list_origin.append(i + 1)

    # 对peak_list_origin做差分，查看两个峰值之间的间隔，保存在数组peak_list_diff中
    peak_list_diff_origin = np.diff(peak_list_origin)

    # 由于初步的峰存在很多噪声，所以会有很多很小的间隔值。
    # 对peak_list_diff排序，则排在前面很小的值为噪声的可能性很大，越往后越有可能是正确的峰。
    # 因此为了预测出真实的峰间距，需要更多的参考后面的数据，即对排序后的peak_list_diff的数据，越往后可信度越高。
    # 因此采用指数加权移动平均模型（EWMA）来预测真实的峰间距

    # 对peak_list_diff_origin排序
    peak_list_diff_origin.sort()

    # 将peak_list_diff_origin数据转换为pandas序列
    peak_list_diff = pd.Series(peak_list_diff_origin)

    # 计算指数加权移动平均
    peak_list_diff_ewma = peak_list_diff.ewm(alpha=0.3).mean()

    # 得到预测间距值
    prediction_peak_spacing = peak_list_diff_ewma.iloc[-1]

    # 原始的峰列表由于噪音问题会存在很多肩峰和重峰的现象。因此我们需要判断哪些峰是同一组的。
    # 将肩峰重峰放在一个数组内，再从这个数组内找出一个峰来代表真实的峰。
    # 创建一个数组peak_list_group用来存放分好组以后的峰，创建一个数组temp_list用来分组时存放同一组的峰
    peak_list_find_peak = peak_list_origin.copy()
    peak_list_group = []
    temp_list = [peak_list_find_peak[0]]

    # 遍历peak_list_find_peak
    for i in range(len(peak_list_find_peak) - 1):

        # 差分，得到峰间距，根据峰间距来判断是否是同一组的峰
        x_diff = peak_list_find_peak[i + 1] - peak_list_find_peak[i]

        # 因为单个二维码水平或者竖直方向最多有26个模块，图像在400*400的空间内，所以正常情况下的最小间距是400/26=15，
        # 再结合我们预测的真实峰间距，若峰间距小于max（3，prediction_peak_spacing/2），则认为是同一组的峰，
        # 直到峰间距大于max（3，prediction_peak_spacing/2）
        if x_diff <= max(8, prediction_peak_spacing / 2):
            temp_list.append(peak_list_find_peak[i + 1])
        else:
            peak_list_group.append(temp_list)
            temp_list = [peak_list_find_peak[i + 1]]

    # 如果是最后一组峰（肩峰或单峰），则直接将其添加到peak_list_group中，避免漏掉尾峰
    if peak_list_find_peak[-1] - temp_list[-1] == 0:
        peak_list_group.append(temp_list)

    # 找出每组峰里最有代表性的峰作为该组峰真正的峰存储到peak_list_group_max，这里采取选择强度最大的峰作为真正的峰。
    peak_list_group_max = []

    # 遍历peak_list_group
    for i in range(len(peak_list_group)):

        # 如果只有一个峰，则直接将其添加到peak_list_group_max中
        if len(peak_list_group[i]) == 1:
            peak_list_group_max.append(int(peak_list_group[i][0]))

        # 如果有多个峰，则找出最大值所对应的j值
        else:
            peak_y_list = []
            for j in range(len(peak_list_group[i])):
                peak_y_list.append(y_sub[int(peak_list_group[i][j])])
            max_y_sub = max(peak_y_list)

            # 如果只有1个max_y_sub，则取最大值所对应的x值
            if peak_y_list.count(max_y_sub) == 1:
                max_y_sub_idx = peak_y_list.index(max_y_sub)
                peak_list_group_max.append(int(peak_list_group[i][max_y_sub_idx]))

            # 如果出现了多个峰的y_sub值相同的情况，则取多个y_sub对应的x值的平均值
            else:
                max_y_sub_idx = [i for i, x in enumerate(peak_y_list) if x == max_y_sub]
                temp = 0
                for j in range(len(max_y_sub_idx)):
                    temp += peak_list_group[i][max_y_sub_idx[j]]
                peak_list_group_max.append(int(temp / len(max_y_sub_idx)))

    # 遍历peak_list_group_max
    for i in range(len(peak_list_group_max) - 1):

        # 计算峰间距
        peak_diff = peak_list_group_max[i + 1] - peak_list_group_max[i]

        # 如果峰间距大于2倍预测峰间距，则需要插入新的峰
        while peak_diff > 2 * prediction_peak_spacing:
            # 在第一个峰间隔预测峰间距处插入一个峰
            peak_list_group_max.insert(i + 1, int(peak_list_group_max[i] + prediction_peak_spacing))

            # 重新计算峰间距
            peak_diff = peak_list_group_max[i + 2] - peak_list_group_max[i + 1]

    return peak_list_group_max


# 定义网格划分函数
def grid_division(peak_list_x, peak_list_y, is_point=False):
    # 存储列信息，即每列的左右端点坐标
    column_list = []

    # 遍历peak_list_x
    for i in range(len(peak_list_x) - 1):
        # 暂存每列的左右端点坐标
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

    # 定义矩阵grid_points，存储网格划分后各网格的四顶点坐标
    grid_points = []

    # 遍历行
    for i in range(len(row_list)):
        # 记录每一行的宽度
        row_width = row_list[i][1] - row_list[i][0]

        # 遍历列
        for j in range(len(column_list)):
            # 记录每一列的宽度
            column_width = column_list[j][1] - column_list[j][0]

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


root = tk.Tk()

root.withdraw()

f_path = filedialog.askopenfilename()

# 采用cv2读取路径为D:\datamatrix\2393的文件夹下有一张名为input.png的PNG图片，将其灰度化
image = cv2.imread(f_path)

# CLAHE对比度增强
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = clahe.apply(image)

# 采用cv2的双边滤波方法，去噪,去除高斯白噪声，尽可能保留边缘信息
k = 5
denoised_image = cv2.bilateralFilter(image, k, 75, 75)

# 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 缩放到400*400的像素空间中
resized_binary_image = cv2.resize(binary_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
resized_denoised_image = cv2.resize(denoised_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)

# 计算每个像素点与右侧一个像素点的灰度差值
Gx = np.abs(np.diff(np.asarray(resized_binary_image), axis=1))

# 统计每一列像素的差值和，最右侧的像素点则与0做差值
col_sum = np.sum(Gx, axis=0)
col_sum = np.append(col_sum, 0)

# 水平方向横纵坐标数组
x_h = np.arange(len(col_sum))
y_h = col_sum

# 计算水平方向分割线
find_peak_list_x = find_peak(y_h)

# 统计每一行像素的差值和，最下方的像素点则与0做差值
Gy = np.abs(np.diff(np.asarray(resized_binary_image), axis=0))
row_sum = np.sum(Gy, axis=1)
row_sum = np.append(row_sum, 0)

# 垂直方向横纵坐标数组
x_v = np.arange(len(row_sum))
y_v = row_sum

# 计算垂直方向分割线
find_peak_list_y = find_peak(y_v)

while len(find_peak_list_x) != len(find_peak_list_y) or len(find_peak_list_x) % 2 == 0 or len(find_peak_list_y) % 2 == 0 and k < 30:
    # 采用cv2的双边滤波方法，去噪,去除高斯白噪声，尽可能保留边缘信息
    k += 2
    denoised_image = cv2.bilateralFilter(image, k, 75, 75)

    # 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
    ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 缩放到400*400的像素空间中
    resized_binary_image = cv2.resize(binary_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    resized_denoised_image = cv2.resize(denoised_image, (1000, 1000), interpolation=cv2.INTER_LINEAR)

    # 计算每个像素点与右侧一个像素点的灰度差值
    Gx = np.abs(np.diff(np.asarray(resized_binary_image), axis=1))

    # 统计每一列像素的差值和，最右侧的像素点则与0做差值
    col_sum = np.sum(Gx, axis=0)
    col_sum = np.append(col_sum, 0)

    # 水平方向横纵坐标数组
    x_h = np.arange(len(col_sum))
    y_h = col_sum

    # 计算水平方向分割线
    find_peak_list_x = find_peak(y_h)

    # 统计每一行像素的差值和，最下方的像素点则与0做差值
    Gy = np.abs(np.diff(np.asarray(resized_binary_image), axis=0))
    row_sum = np.sum(Gy, axis=1)
    row_sum = np.append(row_sum, 0)

    # 垂直方向横纵坐标数组
    x_v = np.arange(len(row_sum))
    y_v = row_sum

    # 计算垂直方向分割线
    find_peak_list_y = find_peak(y_v)

# 得到网格划分后每个网格的四个顶点坐标
grid_points = grid_division(peak_list_x=find_peak_list_x, peak_list_y=find_peak_list_y, is_point=False)

# 创建画布
canvas = np.ones((1220, 1220, 3), dtype=np.uint8) * 255

# 将二值化图像resized_binary_image转换为具有三个通道的灰度图像
gray_image = cv2.cvtColor(resized_binary_image, cv2.COLOR_GRAY2BGR)

# 将二值化图像放置在画布左上角
canvas[0:1000, 0:1000] = gray_image

# 峰值曲线的绘制代码，不要的话注释掉
# # 绘制水平方向曲线，无需拟合
# x_new_h = np.linspace(0, len(x_h) - 1, 1000)
# y_new_h = np.interp(x_new_h, x_h, y_h)

# # 将曲线关于X轴翻转
# y_new_h = np.max(y_new_h) - y_new_h

# # 将坐标范围映射到画布上
# y_new_canvas_h = (y_new_h - np.min(y_new_h)) / (np.max(y_new_h) - np.min(y_new_h)) * 200
# x_new_canvas_h = x_new_h / (len(x_new_h) - 1) * 1000

# for i in range(len(x_new_canvas_h) - 1):
#     p1 = (int(x_new_canvas_h[i]), 1200 - int(y_new_canvas_h[i]))
#     p2 = (int(x_new_canvas_h[i + 1]), 1200 - int(y_new_canvas_h[i + 1]))
#     cv2.line(canvas, p1, p2, (0, 0, 255), 2)

# # 绘制垂直方向曲线，无需拟合
# x_new_v = np.linspace(0, len(x_v) - 1, 1000)
# y_new_v = np.interp(x_new_v, x_v, y_v)

# # 将曲线关于X轴翻转
# y_new_v = np.max(y_new_v) - y_new_v

# # 将坐标范围映射到画布上
# y_new_canvas_v = (y_new_v - np.min(y_new_v)) / (np.max(y_new_v) - np.min(y_new_v)) * 200
# x_new_canvas_v = x_new_v / (len(x_new_v) - 1) * 1000

# for i in range(len(x_new_canvas_v) - 1):
#     p1 = (1200 - (int(y_new_canvas_v[i])), int(x_new_canvas_v[i]))
#     p2 = (1200 - (int(y_new_canvas_v[i + 1])), int(x_new_canvas_v[i + 1]))
#     cv2.line(canvas, p1, p2, (0, 255, 0), 2)

# 遍历peak_x_list_find_peak，将获取的值作竖直垂线绘制在画布上
for i in range(len(find_peak_list_x)):
    # p1 = (int(x_new_canvas_h[find_peak_list_x[i]]), 0)
    # p2 = (int(x_new_canvas_h[find_peak_list_x[i]]), 1200)
    p1 = (int(find_peak_list_x[i]), 0)
    p2 = (int(find_peak_list_x[i]), 1200)
    # 25.2.10 不要峰值曲线
    cv2.line(canvas, p1, p2, (204, 209, 72), 2)

# 遍历peak_y_list_find_peak，将获取的值作水平垂线绘制在画布上
for i in range(len(find_peak_list_y)):
    p1 = (0, find_peak_list_y[i])
    p2 = (1200, find_peak_list_y[i])
    cv2.line(canvas, p1, p2, (204, 209, 72), 2)

# 遍历每个网格的四个顶点坐标
for point_set in grid_points:
    # 遍历每个顶点坐标
    for point in point_set:
        # 在画布上绘制红色圆点
        cv2.circle(canvas, point, radius=5, color=(0, 0, 255), thickness=-1)

# 保存路径
save_path_dir = os.path.join(os.path.dirname(f_path), 'output')
os.makedirs(save_path_dir, exist_ok=True)
save_path = os.path.join(save_path_dir, os.path.basename(f_path))

# 保存画布
cv2.imwrite(save_path, canvas)
