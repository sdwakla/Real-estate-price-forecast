# -*- coding: utf-8 -*-
# @Time : 2023/3/27 22:02
# @Author :JI ZIAO
# @File : demo.py
# @contact:51095836

import joblib
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.datasets import load_boston
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)




# TODO： visualization & Feature Engineering
# ----------------------- 目标变量可视化 --------------------
def show_datainfo(data):
    print(data)
    print(data.head(5))
    # Dimension of the dataset
    print(np.shape(data))
    # Let's summarize the data to see the distribution of data
    print(data.describe())

def show_price(data):
    # plt.hist(data.PRICE, bins=50)
    # plt.xlabel('Price in $1000s')
    # plt.ylabel('Number of houses')
    # plt.show()
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(data['PRICE'], kde=True, bins=30)
    plt.title('PRICE Histogram')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

    # 绘制QQ图
    plt.figure(figsize=(10, 6))
    stats.probplot(data['PRICE'], plot=plt)
    plt.title('PRICE QQ-Plot')
    plt.show()


# ----------------- 自变量特征变量的分布情况 -------------------
def show_vir(data):
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k, v in data.items():
        sns.distplot(v, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()


# --------------- 自变量相互关系研究 --------------------
# 选择部分
def show_corr(data):
    # sns.pairplot(data[["LSTAT", "RM", "PIRATIO", "PRICE"]])
    sns.pairplot(data)
    plt.show()


# TODO：特征工程增加变量纬度
# --------------------------- 热力图可视化 -----------------
def show_heatMap(data):
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr().abs(), annot=True)
    # plt.title('Correlation of Features')
    plt.show()
    # 通过上面分析进行筛选
    column_sels = ['LSTAT', 'RM', 'PIRATIO']
    # 可视化查看了部分变量之间的分布，可解释性不强：
    # 新思路使用相关系数之后和PRICE进行线性可视化
    # Let's scale the columns before plotting them against MEDV
    min_max_scaler = preprocessing.MinMaxScaler()
    x = data.loc[:, column_sels]
    y = data['PRICE']
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for i, k in enumerate(column_sels):
        sns.regplot(y=y, x=x[k], ax=axs[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()



# -------------------- 检测和处理离群点 ------------------
# 先使用box plot来展示一下异常值情况，然后使用四分卫点进行剔除
def show_box(data):
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 15))
    index = 0
    axs = axs.flatten()
    for k, v in data.items():
        # axs[index].set_xlabel("X label for {}".format(k))
        sns.boxplot(y=k, data=data, ax=axs[index])
        index += 1
    plt.subplots_adjust(hspace=0.4)
    # plt.suptitle("Title for all subplots")
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()



def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1  # 四分位距（IQR）
        outlier_border = 1.5 * IQR  # 离群点边界
        outlier_index_list = df[(df[col] < Q1 - outlier_border) | (df[col] > Q3 + outlier_border)].index  # 离群点的索引
        outlier_indices.extend(outlier_index_list)  # 将索引添加到列表中
    # 找到出现超过n次的离群点
    outlier_indices = [x for x in outlier_indices if outlier_indices.count(x) > n]
    return outlier_indices


def pre_outliers(data):
    # 检测并处理离群点
    outliers = detect_outliers(data, 2, data.columns)
    data = data.drop(outliers, axis=0).reset_index(drop=True)
    return data


def pre_scaled(X):
    # ----------------------- 特征缩放 --------------------
    # 创建标准化器
    scaler = StandardScaler()

    # 对特征变量进行标准化
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # ---------------- 构建新特征----------------
    # 交互特征、缩放特征、多项式特征
    X_scaled['LSTAT_RM'] = X_scaled['LSTAT'] * X_scaled['RM']
    X_scaled['TAX_PIRATIO'] = X_scaled['TAX'] * X_scaled['PIRATIO']
    X_scaled['TAXRM'] = X_scaled['TAX'] / X_scaled['RM']
    X_scaled['LSTAT_SQ'] = X_scaled['LSTAT'] ** 2
    X_scaled['RM_LOG'] = np.log(X_scaled['RM'] + 1e-10)
    X_scaled['DIS_LOG'] = np.log(X_scaled['DIS'] + 1e-10)
    X_scaled['NOX_SQRT'] = np.sqrt(X_scaled['NOX'])
    X_scaled['AGE_LOG'] = np.log(X_scaled['AGE'] + 1e-10)
    X_scaled['INDUS_SQRT'] = np.sqrt(X_scaled['INDUS'])
    X_scaled['ZN_SQRT'] = np.sqrt(X_scaled['ZN'])
    X_scaled['RAD_LOG'] = np.log(X_scaled['RAD'] + 1e-10)
    X_scaled['B_LOG'] = np.log(X_scaled['B'] + 1e-10)
    X_scaled['PIRATIO_SQ'] = X_scaled['PIRATIO'] ** 2
    X_scaled['TAX_LOG'] = np.log(X_scaled['TAX'] + 1e-10)
    # X_scaled['PRICE_LOG'] = np.log(X_scaled['PRICE'])

    imputer = SimpleImputer(strategy='mean')
    # fit the imputer to the data
    imputer.fit(X_scaled)
    # transform the data
    X_imputed = imputer.transform(X_scaled)
    X_imputed = pd.DataFrame(X_imputed, columns=X_scaled.columns)

    # 多项式特征
    # poly = PolynomialFeatures(degree=2)
    # X_poly = poly.fit_transform(X_imputed)
    return X_imputed


# TODO:特征筛选和降维（PCA）
def pre_pca(data):
    """
    效果不理想，可以用来做数据挖掘
    更加适用的是相关系数选择关键
    :param data:
    :return:
    """
    X_imputed = data
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_imputed)
    feature_names = X_imputed.columns
    # Get the weights of each feature in the principal components
    component_weights = pca.components_
    # Print the names of the features that have the highest absolute weight in each principal component
    for i, component in enumerate(component_weights):
        top_features = feature_names[abs(component).argsort()[::-1][:5]]
        print(f"Top features in component {i + 1}: {', '.join(top_features)}")
    return X_pca


def pre_hearmap_topfeatures(data):
    correlation_matrix = data.corr().round(2)
    correlation_list = correlation_matrix['PRICE'].sort_values(ascending=False)  # 将相关性按照大小排序
    top_features = correlation_list[abs(correlation_list) > 0.3].index.tolist()  # 选择相关性最强的前几个特征
    return data[top_features]




# TODO：Models
def exp1_single_performance_improvement_lr():
    from sklearn.datasets import load_boston
    # Load the dataset
    boston = load_boston()
    X, y = boston.data, boston.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform feature engineering (Polynomial features)
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Remove outliers
    z_scores = np.abs(stats.zscore(X_train_scaled))
    outlier_threshold = 3
    X_train_no_outliers = X_train_scaled[(z_scores < outlier_threshold).all(axis=1)]
    y_train_no_outliers = y_train[(z_scores < outlier_threshold).all(axis=1)]

    # Define a function to evaluate a model
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        return mse_train, mse_test

    # Define the models
    baseline = LinearRegression()

    # Model A: Hyperparameter tuning
    ridge_params = {'alpha': np.logspace(-3, 3, 7)}
    ridge = Ridge(random_state=42)
    ridge_cv = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')
    # Fit the GridSearchCV object
    ridge_cv.fit(X_train_scaled, y_train)
    # Extract the best hyperparameters
    best_alpha = ridge_cv.best_params_['alpha']
    # Create and fit the Ridge model with the best hyperparameters
    ridge_best = Ridge(alpha=best_alpha, random_state=42)
    ridge_best.fit(X_train_scaled, y_train)

    # Model B: Feature engineering
    model_b = LinearRegression()

    # Model C: Outlier removal
    model_c = LinearRegression()

    # Model D: Hyperparameter tuning + Feature engineering + Outlier removal
    model_d = Ridge(alpha=ridge_cv.best_params_['alpha'], random_state=42)

    # Evaluate the models
    mse_baseline_train, mse_baseline_test = evaluate_model(baseline, X_train_scaled, X_test_scaled, y_train, y_test)
    mse_model_a_train, mse_model_a_test = evaluate_model(ridge_best, X_train_scaled, X_test_scaled, y_train, y_test)
    mse_model_b_train, mse_model_b_test = evaluate_model(model_b, X_train_poly, X_test_poly, y_train, y_test)
    mse_model_c_train, mse_model_c_test = evaluate_model(model_c, X_train_no_outliers, X_test_scaled,
                                                         y_train_no_outliers, y_test)
    mse_model_d_train, mse_model_d_test = evaluate_model(model_d, X_train_no_outliers, X_test_scaled,
                                                         y_train_no_outliers, y_test)

    # Print the results
    print("Model\t\t\tTrain MSE\tTest MSE")
    print("Baseline\t\t{:.4f}\t\t{:.4f}".format(mse_baseline_train, mse_baseline_test))
    print("Model A (Hyperparameters)\t{:.4f}\t\t{:.4f}".format(mse_model_a_train, mse_model_a_test))
    print("Model B (Feature Eng.)\t\t{:.4f}\t\t{:.4f}".format(mse_model_b_train, mse_model_b_test))
    print("Model C (Outlier Removal)\t{:.4f}\t\t{:.4f}".format(mse_model_c_train, mse_model_c_test))
    print("Model D (Combined)\t\t{:.4f}\t\t{:.4f}".format(mse_model_d_train, mse_model_d_test))


def exp1_single_performance_improvement_gdbt(X,y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 基本GradientBoostingRegressor模型
    gb_base = GradientBoostingRegressor(random_state=42)
    gb_base.fit(X_train, y_train)
    y_pred_base = gb_base.predict(X_test)
    mse_base = mean_squared_error(y_test, y_pred_base)
    print("Base model MSE:", mse_base)

    # 超参数调整
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(gb_base, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best parameters for GradientBoostingRegressor:", best_params)

    # 使用最佳超参数训练模型
    gb_tuned = GradientBoostingRegressor(**best_params, random_state=42)
    gb_tuned.fit(X_train, y_train)
    y_pred_tuned = gb_tuned.predict(X_test)
    mse_tuned = mean_squared_error(y_test, y_pred_tuned)
    print("Tuned model MSE:", mse_tuned)

    # 特征工程：标准化
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', gb_tuned)
    ])

    pipeline.fit(X_train, y_train)
    y_pred_scaled = pipeline.predict(X_test)
    mse_scaled = mean_squared_error(y_test, y_pred_scaled)
    print("Scaled model MSE:", mse_scaled)

    # 异常值处理：删除异常值
    data_outliers = pre_outliers(data) # 剔除异常值
    X = data_outliers.drop('PRICE', axis=1)
    y = data_outliers['PRICE']

    X_imputed = pre_scaled(X) # 数据标准化+特征工程
    X_pca = pre_pca(X_imputed) # 数据降维
    X_filtered = X_pca
    y_filtered = data_outliers['PRICE']
    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered,
                                                                                            test_size=0.2,
                                                                                            random_state=42)

    # 使用处理后的数据训练模型
    pipeline.fit(X_train_filtered, y_train_filtered)
    y_pred_filtered = pipeline.predict(X_test_filtered)
    mse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered)
    print("Filtered model MSE:", mse_filtered)


def exp2_multimodal_compare(x, y):
    """

    :param x:
    :param y:
    :return:
    """

    flag1,flag2 = 0,0 # 控制调优和保存模型
    scores_map = {}
    kf = KFold(n_splits=10)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    l_regression = linear_model.LinearRegression()
    l_regression.fit(x_scaled, y)
    scores = abs(cross_val_score(l_regression, x_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    print("LinearRegression MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    scores_map['LinearRegression'] = scores

    l_ridge = linear_model.Ridge()
    l_ridge.fit(x_scaled, y)
    scores = abs(cross_val_score(l_ridge, x_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    print("Ridge MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    scores_map['Ridge'] = scores

    # Lets try polinomial regression with L2 with degree for the best fit

    # for degree in range(2, 6):
    #    model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.Ridge())
    #    scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    #    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    # model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
    # scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    # print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    # scores_map['PolyRidge'] = scores
    """
    The Liner Regression with and without L2 regularization does not make significant difference is MSE score. 
    However polynomial regression with degree=3 has a better MSE. Let's try some non prametric regression techniques: SVR with kernal rbf, DecisionTreeRegressor, KNeighborsRegressor etc.
    """

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # svr_rbf.fit(x_scaled, y)
    if flag1:
        grid_sv = GridSearchCV(svr_rbf, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        # print("Best classifier :", grid_sv.best_estimator_)
        best_mode_svr_rbf = grid_sv.best_estimator_

    scores = abs(cross_val_score(svr_rbf, x_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    print("SVR MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    scores_map['SVR'] = scores

    desc_tr = DecisionTreeRegressor(max_depth=5)
    # desc_tr.fit(x_scaled, y)
    if flag1:
        grid_sv = GridSearchCV(desc_tr, cv=kf, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        # print("Best classifier :", grid_sv.best_estimator_)
        best_mode_desc_tr = grid_sv.best_estimator_

    scores = abs(cross_val_score(desc_tr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    print("DecisionTreeRegressor MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    scores_map['DecisionTreeRegressor'] = scores

    knn = KNeighborsRegressor(n_neighbors=7)
    # knn.fit(x_scaled, y)
    if flag1:
        grid_sv = GridSearchCV(knn, cv=kf, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        # print("Best classifier :", grid_sv.best_estimator_)
        best_mode_knn = grid_sv.best_estimator_

    scores = abs(cross_val_score(knn, x_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    scores_map['KNeighborsRegressor'] = scores
    print("KNN MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    """
    Compared to three models which are shosen through grid search, SVR performes better. Let's try an ensemble method finally.
    """

    gbr = GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=2, min_samples_leaf=5, min_samples_split=2,
                                    n_estimators=100, random_state=30)
    if flag1:
        param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05, 0.02], 'max_depth': [2, 4, 6],
                      'min_samples_leaf': [3, 5, 9]}
        grid_sv = GridSearchCV(gbr, cv=kf, param_grid=param_grid, scoring='neg_mean_squared_error')
        grid_sv.fit(x_scaled, y)
        # print("Best classifier :", grid_sv.best_estimator_)
        best_mode_gbr = grid_sv.best_estimator_

    scores = abs(cross_val_score(gbr, x_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    print("GradientBoostingRegressor MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    scores_map['GradientBoostingRegressor'] = scores

    # scores_map.pop('PolyRidge')  # 性能异常

    # Let's plot k-fold results to see which model has better distribution of results. Let's have a look at the MSE distribution of these models with k-fold=10

    plt.figure(figsize=(20, 10))
    scores_map = pd.DataFrame(scores_map)
    sns.boxplot(data=scores_map)
    plt.show()
    # The models SVR and GradientBoostingRegressor show better performance.

    # 需要保存的模型
    if flag1 and flag2:
        model1 = l_regression
        model2 = l_ridge
        model3 = best_mode_svr_rbf
        model4 = best_mode_desc_tr
        model5 = best_mode_knn
        model6 = best_mode_gbr
        joblib.dump(model1, 'boston_house_price_model_l_regression.joblib')
        joblib.dump(model2, 'boston_house_price_model_l_ridge.joblib')
        joblib.dump(model3, 'boston_house_price_model_svr_rbf.joblib')
        joblib.dump(model4, 'boston_house_price_model_desc_tr.joblib')
        joblib.dump(model5, 'boston_house_price_model_knn.joblib')
        joblib.dump(model6, 'boston_house_price_model_gbr.joblib')



if __name__ == '__main__':
    # ------ 数据读取 ----------
    data = pd.read_csv('train_dataset.csv')


    # ------ 可视化 -------
    # show_datainfo(data)
    # show_price(data)
    # show_vir(data)
    # show_corr(data)
    # show_heatMap(data)
    # show_box(data)

    # ----- 数据集预处理 --------
    # # 加载波士顿房价数据集
    # boston = load_boston()
    # X = boston.data
    # y = boston.target

    # 处理数据的流程：剔除异常值-》数据标准化-》特征工程-》降维度(融合到exp1当中)
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']
    # data_outliers = pre_outliers(data) # 剔除异常值
    # X = data_outliers.drop('PRICE', axis=1)
    # y = data_outliers['PRICE']
    #
    # X_imputed = pre_scaled(X) # 数据标准化+特征工程
    # X_pca = pre_pca(X_imputed) # 数据降维
    # X, y = X_pca, data['PRICE']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    exp1_single_performance_improvement_lr()
    # exp1_single_performance_improvement_gdbt(X, y)
    exp2_multimodal_compare(x=X, y=y)