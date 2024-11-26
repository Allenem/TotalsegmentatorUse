import nibabel as nib
import numpy as np
from scipy.ndimage import label
from sklearn.cluster import KMeans
from scipy.ndimage import center_of_mass

# 1. 加载 NIfTI 图像
input_path = './label/output1/vertebrae.nii.gz'  # 输入 NIfTI 文件路径
img = nib.load(input_path)
data = img.get_fdata()  # 获取图像数据为 NumPy 数组

# 2. 确保图像值为 0 和 1
binary_data = (data > 0).astype(np.int32)

# 3. 对值为1的区域进行连通区域标记
labeled_array, num_features = label(binary_data)
print(f"共检测到 {num_features} 个连通区域")
print(f'连通区域标记后的数组为：{labeled_array}, shape：{labeled_array.shape}，min:{labeled_array.min()}, max:{labeled_array.max()}')
output_path = f'./label/output1/vertebrae{num_features}.nii.gz'  # 输出 NIfTI 文件路径

# 4. 获取所有连通区域的重心
# centers = measurements.center_of_mass(binary_data, labeled_array, range(1, num_features + 1))
centers = center_of_mass(binary_data, labeled_array, range(1, num_features + 1))
centers_array = np.array(centers)

# 5. 聚类为 num_features 类
kmeans = KMeans(n_clusters=num_features, random_state=0)
kmeans.fit(centers_array)
labels = kmeans.labels_
print(f"聚类后的标签为：{labels} len:{len(labels)}")

# 6. 根据中心的z轴位置排序
sorted_indices = np.argsort(-centers_array[:, 2])
sorted_labels = np.zeros_like(labels)
for i, index in enumerate(sorted_indices):
    sorted_labels[labels == index] = i
print(f"聚类后的中心点为：{centers_array}")
print(f"按照z轴位置排序后的索引为：{sorted_indices}")
print(f"按照z轴位置排序后的标签为：{sorted_labels} len:{len(sorted_labels)}")

# 7. 创建新的分割图像，将每个聚类的值替换为对应的类标签
clustered_data = np.zeros_like(binary_data)
# for cluster_label, center in zip(sorted_labels, range(1, num_features + 1)):
#     clustered_data[labeled_array == center] = cluster_label + 1
# for cluster_label, center in zip(sorted_indices, range(1, num_features + 1)):
#     clustered_data[labeled_array == center] = cluster_label + 1
#     print(f"cluster_label:{cluster_label}, center:{center}")
for oldlabel, newlabel in zip(labels, sorted_labels):
    clustered_data[labeled_array == (oldlabel+1)] = newlabel + 1
    print(f"old:{oldlabel+1}, new:{newlabel+1}")

# 8. 保存新的 NIfTI 图像
new_img = nib.Nifti1Image(clustered_data, img.affine, img.header)
nib.save(new_img, output_path)

print(f"聚类后的图像已保存至: {output_path}")