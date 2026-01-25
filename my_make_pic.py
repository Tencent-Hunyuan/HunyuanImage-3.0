import matplotlib.pyplot as plt
import numpy as np

# 1. 数据准备
models = ['Nano Banana Pro', 'Seedream-4.5', 'Qwen-Image-Edit-2511']
win_rates = np.array([[-1.43, 7.13, 23.59], [-2.60, 10.48, 34.71]])

# 2. 配色方案设置
# 为负值选择冷色调，为正值选择暖色调/亮色调，确保区分明显且美观
# Nano Banana (负值，冷蓝灰), Seedream4.5 (正值，青绿), QwenImage (大正值，活力橙)
# colors = ['#A8DF8E', '#FFD8DF', '#FFA239']
# 两列数据使用两种颜色，确保区分明显且美观
colors = ['#A8DF8E', '#FFA239']
label1 = "Internal R&D Testset"
label2 = "User Preference Testset"

# 设置画布大小和分辨率
fig, ax = plt.subplots(figsize=(9, 6), dpi=100)

# 3. 绘制柱状图（每个 model 两列）
# zorder=3 确保柱子在网格线图层之上
x = np.arange(len(models))
bar_width = 0.34
bars_1 = ax.bar(x - bar_width / 2, win_rates[0], color=colors[0], width=bar_width,
                zorder=3, edgecolor='white', linewidth=0.8, label=label1)
bars_2 = ax.bar(x + bar_width / 2, win_rates[1], color=colors[1], width=bar_width,
                zorder=3, edgecolor='white', linewidth=0.8, label=label2)

# 4. 添加重要的基准线 (y=0)
# 这条线对于展示负值至关重要，使用深灰色突出显示
ax.axhline(0, color='#333333', linewidth=1.2, linestyle='-', zorder=2)

# 5. 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()

        # 根据正负值决定标签的位置和垂直对齐方式
        if height >= 0:
            # 正值：放在柱子上方，底部对齐
            label_y_pos = height + 0.8
            va_align = 'bottom'
            color_text = 'black'
            # color_text = bar.get_facecolor()
        else:
            # 负值：放在柱子下方，顶部对齐
            label_y_pos = height - 1.2
            va_align = 'top'
            color_text = 'black'
            # 负值标签用柱子颜色
            # color_text = bar.get_facecolor()

        ax.text(bar.get_x() + bar.get_width() / 2.,  # X坐标：柱子中心
                label_y_pos,                         # Y坐标
                f'{height:.2f}%',                    # 显示两位小数，保留尾部 0
                ha='center',                         # 水平居中
                va=va_align,                         # 垂直对齐方式动态调整
                fontsize=11, fontweight='bold', color=color_text)

add_value_labels(bars_1)
add_value_labels(bars_2)

# 6. 图表美化与标注
# 设置标题和坐标轴标签
ax.set_title('HunyuanImage3.0-Instruct Win Rate (GSB)', fontsize=16, fontweight='bold', pad=25, color='#222222')
ax.set_ylabel('Win Rate', fontsize=12, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(frameon=False)

# 隐藏顶部和右侧的边框，使图表更清爽
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 隐藏底部边框，因为我们已经手动添加了更明显的 y=0 基准线
ax.spines['bottom'].set_visible(False)

# 添加Y轴网格线，增加可读性 (zorder=0 放在最底层)
ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)

# 调整X轴标签的样式
ax.tick_params(axis='x', labelsize=11, length=0, pad=0) # length=0 隐藏刻度短线
# 某些环境下 pad 不明显，手动把标签往上挪一点
for label in ax.get_xticklabels():
    label.set_y(-0.001)

# 动态调整Y轴范围，确保标签不会被画布边缘遮挡
ax.set_ylim(win_rates.min() - 5, win_rates.max() + 7)

# 自动调整布局并显示
plt.tight_layout()
plt.show()
plt.savefig("assets/gsb_instruct.png", bbox_inches="tight")