# -*- coding: utf-8 -*- 
# @Time : 2023/4/5 15:41 
# @Author :JI ZIAO
# @File : demo3.py 
# @contact: 51095836


import joblib
import numpy as np


def load_model(model_name):
    return joblib.load(model_name)


# 用于将用户输入的字符串转换为浮点数列表
def parse_input(input_str):
    return [float(x) for x in input_str.split()]


available_models = {
    "1": "boston_house_price_model_l_regression.joblib",
    "2": "boston_house_price_model_l_ridge.joblib",
    "3": "boston_house_price_model_svr_rbf.joblib",
    "4": "boston_house_price_model_desc_tr.joblib",
    "5": "boston_house_price_model_knn.joblib",
    "6": "boston_house_price_model_gbr.joblib"
}

while True:
    print("\nAvailable models:")
    print("1. Boston House Price Model")
    print("0. Exit")

    # 选择模型
    model_choice = input("Select a model by entering its number or '0' to exit: ")

    if model_choice == "0":
        print("Exiting...")
        break
    elif model_choice in available_models:
        model_filename = available_models[model_choice]
        model = load_model(model_filename)

        try:
            input_str = input("Enter the input data as a space-separated list of numbers: ")
            input_data = parse_input(input_str)
            input_array = np.array([input_data])

            prediction = model.predict(input_array)
            print("Prediction:", prediction[0])
        except Exception as e:
            print("Error: Unable to parse input. Please ensure the input data is formatted correctly.")
    else:
        print("Invalid selection. Please try again.")


"""
测试用例：
0.17004 12.5 7.87 0 0.524 6.004 85.9 6.5921 5 311 15.2 386.71 17.1 
18.9

0.95577 0 8.14 0 0.538 6.047 88.8 4.4534 4 307 21 306.38 17.28
14.8
"""