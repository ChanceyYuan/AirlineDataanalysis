'''航空客户分类(基于k-means)'''

'''数据清洗'''
import pandas as pd

#读取数据
data = pd.read_excel(r'F:\python_work\AirlinedataProcessing\i_nuc.xls',
                     index_col = 'Id',
                     sheetname = 'Sheet2')                                       #index_col设置index


#设置聚类
outputfile = r'F:\python_work\AirlinedataProcessing\data_type.xls'               #结果保存的文件名
k = 3                                                                            #设置聚类的类别
iteration = 500                                                                  #设置据类的最大循环次数



'''标准化处理'''
bzfile = r'F:\python_work\AirlinedataProcessing\bzfile.xls'                         #标准化文件保存路径
data_bz = 1.0*(data - data.mean())/data.std()                                    #数据标准化
data_bz.to_excel(bzfile,index = False)                                           #保存数据到文件



'''K-Means算法聚类分析'''
from sklearn.cluster import KMeans

model = KMeans(n_clusters = k ,                                                  #簇的个数，拟聚合成k类
               n_jobs = 4,                                                       #用几核cpu进行训练
               max_iter = iteration)                                             #最大迭代次数

#开始聚类
model.fit(data_bz)                                          
#简单打印结果
r1 = pd.Series(model.labels_).value_counts()                                     #统计各个类别的树木
r2 = pd.DataFrame(model.cluster_centers_)                                        #找出聚类中心的矩阵
r = pd.concat([r2,r1],axis = 1)
r.columns = list(data.columns) + [u'类别数目']
#详细打印结果
r = pd.concat([data,pd.Series(model.labels_,index =data.index)],axis = 1)
r.columns = list(data.columns) + [u'聚类类别']
#保存结果
r.to_excel(outputfile)

#自定义作图函数
def density_plot(data):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei']              #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False              #正常显示负号
    p = data.plot(kind = 'kde',
                  linewidth = 2,
                  subplots = True,
                  sharex = False)
    [p[i].set_ylabel(u'密度') for i in range(k)]
    plt.xlabel('分群%s'%(i+1))
    plt.legend()
    return plt

pic_output = r'F:\python_work\AirlinedataProcessing\pd_'

for i in range(k):
    density_plot(data[r[u'聚类类别']==i]).savefig(u'%s%s.png'%(pic_output,i))


'''主函数KMeans'''
#sklearn.cluster.KMeans(
#    n_clusters = 8,
#    n_init = 10,
#    max_iter = 300,
#    tol = 0.0001,
#    precompute_distances = 'auto',
#    verbose = 0,
#    random_state = None,
#    copy_x = True,
#    n_jobs = 1,
#    algorithm = 'auto'
#    )