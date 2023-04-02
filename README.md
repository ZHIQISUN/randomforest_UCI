# randomforest_UCI
# -*- coding: utf-8 -*-
def plot_roc_curves_for_models(model, splits):
    train_x, test_x, train_y, test_y = splits
    fig, ax = plt.subplots()
    test_prob = model.predict_proba(test_x)[:, 1]
    train_prob = model.predict_proba(train_x)[:, 1]
    fpr_, tpr_, thresholds_ = metrics.roc_curve(train_y, train_prob)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
    auc_ = roc_auc_score(train_y, train_prob)
    auc = roc_auc_score(test_y, test_prob)
    ax.plot(fpr_, tpr_, label=(f"RF_train_AUC_area = {auc_:.2f}"))
    ax.plot(fpr, tpr, label=(f"RF_test_AUC_area = {auc:.2f}"))
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic For RandomForest")
    ax.legend(loc="lower right")
    return fig
def model_performance(model, test_x, test_y, index_indata, verbose=True):
    test_prob = model.predict_proba(test_x)[:, 1]
    test_pred = model.predict(test_x)
    accuracy = accuracy_score(test_y, test_pred)
    precision = precision_score(test_y, test_pred)
    sens = recall_score(test_y, test_pred)
    spec = recall_score(test_y, test_pred, pos_label=0)
    auc = roc_auc_score(test_y, test_prob)
    mcc = matthews_corrcoef(test_y, test_pred)
    print("真实标签：")
    for i in range(len(test_y)):
      print(test_y[i], end = ",")
    print("\n预测标签：")
    for i in range(len(test_y)):
      print(test_pred[i], end = ",")
    print()
    if verbose:
        print(f"准确度: {accuracy:.2f}")
        print(f"精确度: {precision:.2f}")
        print(f"敏感性/召回率: {sens:.2f}")
        print(f"特异性: {spec:.2f}")
        print(f"马修斯相关系数: {mcc:.2f}")
        print(f"AUC: {auc:.2f}")
    return accuracy, precision, sens, spec, mcc, auc
def model_training_and_validation(model, name, splits, verbose=True):
    train_x, test_x, train_y, test_y = splits
    model.fit(train_x, train_y)
    if verbose:
        print("训练集：")
    model_performance(model, train_x, train_y, verbose)
    if verbose:
        print("测试集：")
    accuracy, precision, sens, spec, mcc, auc = model_performance(model, test_x, test_y, verbose)
    
    return accuracy, precision, sens, spec, mcc, auc
import PySimpleGUI as sg
layout = [
  [sg.T("-------------------",font = ("楷体",30), auto_size_text = True, justification = "center")],
  [sg.T("随机森林分类预测模块",font = ("楷体",30), auto_size_text = True, justification = "center")],
  [sg.T("-------------------",font = ("楷体",30), auto_size_text = True, justification = "center")],
  [sg.B("导入数据",font = ("楷体",20), auto_size_button = True, pad = (30,30)),sg.B("数据可视化",font = ("楷体",20), auto_size_button = True, pad = (30,30))],
  [sg.B("数据预处理",font = ("楷体",20), auto_size_button = True, pad = (30,30)),sg.B("参数寻优",font = ("楷体",20), auto_size_button = True, pad = (30,30))],
  [sg.B("模型效果评估",font = ("楷体",20), auto_size_button = True, pad = (30,30)),sg.B("退出",font = ("楷体",20), auto_size_button = True, pad = (60,30))]
]
window = sg.Window("随机森林分类UCI乳腺癌数据",layout)
while True:
  event,values = window.read()
  print(event)
  print(values)
  if event == None:
    break
  if event == "导入数据":
    layout1 = [
    [sg.T("使用UCI数据集中的diabetes数据集，为二分类数据",font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("基于sklearn的randomforest进行建模",font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("数据导入成功，请进行数据可视化和数据预处理", font = ("楷体",20), auto_size_text = True, justification = "center")],
    ]
    window1 = sg.Window("导入数据...", layout1)
    import numpy as np
    from sklearn import datasets, preprocessing, metrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef, roc_auc_score
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.decomposition import PCA
    Data = datasets.load_breast_cancer()
    X = np.array(Data.data)  # 数据集数据
    Y = np.array(Data.target)  # 数据集标签
    while True:
      event1,values1 = window1.read()
      if event1 == None:
        break
  if event == "数据可视化":
    marker = ['*','o']
    color = ['b', 'g']
    labels = ['Zero', 'One']
    pca=PCA(n_components=2)
    pca.fit(X)
    result = pca.transform(X)
    plt.figure(figsize=(8, 5))
    plt.title('PCA process for breast_cancer_feature')
    plt.ylabel("PCA2")
    plt.xlabel("PCA1")
    for i in range(result.shape[1]):
      plt.scatter(result[Y == i,0], result[Y == i,1], c = color[i], marker = marker[i], label = labels[i])
    plt.legend(loc='upper right')
    plt.show()
  if event == "数据预处理":
    layout2 = [
    [sg.T("scikit-learn随机划分,比例为4:1",font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("数据压缩，使输入数据标准化",font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("请进行参数寻优", font = ("楷体",20), auto_size_text = True, justification = "center")],
    ]
    window2 = sg.Window("导入数据...", layout2)
    (static_train_x,static_test_x,static_train_y,static_test_y) = train_test_split(X,Y, test_size=0.2)
    print(len(static_train_x),len(static_test_x))
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(static_train_x)
    static_test_x = scaler.transform(static_test_x) 
    static_train_x = scaler.transform(static_train_x) 
    X = scaler.transform(X) 
    splits= [static_train_x, static_test_x, static_train_y, static_test_y]
    while True:
      event2,values2 = window2.read()
      if event2 == None:
        break
  if event == "参数寻优":
    grid_dict = {"max_leaf_nodes":[i for i in range(10,30,2)], "n_estimators":[i for i in range(10,50,5)]}
    rf = RandomForestClassifier()
    rf_grid = GridSearchCV(estimator = rf, param_grid = grid_dict, scoring="accuracy", cv = 4, verbose=0, n_jobs=-1)
    rf_grid.fit(static_train_x, static_train_y)
    layout3 = [
    [sg.T("使用网格搜索法对随机森林的树个数和最大叶节点数进行参数寻优：",font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("将训练集随机划分为训练集和验证集，比例为4:1，使用4折交叉验证",font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("为399个候选参数每个拟合4次，共1596次...", font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T("参数寻优结果如下，最好参数和最佳4折交叉验证得分为：", font = ("楷体",20), auto_size_text = True, justification = "center")],
    [sg.T(f"{rf_grid.best_params_}, {rf_grid.best_score_}", font = ("楷体",20), auto_size_text = True, justification = "center")],
    ]
    window3 = sg.Window("导入数据...", layout3)
    while True:
      event3,values3 = window3.read()
      if event3 == None:
        break
  if event == "模型效果评估":
    model_RF = RandomForestClassifier(**rf_grid.best_params_)
    accuracy, precision, sens, spec, mcc, auc = model_training_and_validation(model_RF, "RF", splits)
    sg.popup(f"测试集结果:\n准确度: {accuracy:.2f}\n精确度: {precision:.2f}\n敏感性/召回率: {sens:.2f}\n特异性: {spec:.2f}\n马修斯相关系数: {mcc:.2f}\nAUC: {auc:.2f}")
    plot_roc_curves_for_models(model_RF, splits)
    plt.show()
  if event == "退出":
    break
window.close()
