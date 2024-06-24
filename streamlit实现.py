import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf 
import random
import pandas as pd
from surprise.dump import load
from fastai.vision.all import *
import pathlib
import sys

# 根据不同的操作系统设置正确的pathlib.Path
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath


# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path,"classify.pkl")
# 加载图像识别模型
image_model = load_learner(model_path)

# 饰品图片数据库
饰品图片数据库 = {}

# 饰品文件夹路径
饰品文件夹路径 = "C:\\Users\\郑蕊\\Desktop\\饰品推荐系统\\饰品的十种不同风格"

# 读取文件夹中的图片
饰品类型 = os.listdir(饰品文件夹路径)
for 类型 in 饰品类型:
    类型路径 = os.path.join(饰品文件夹路径, 类型)
    饰品图片数据库[类型] = [os.path.join(类型路径, img) for img in os.listdir(类型路径)]

# 图像识别函数
def predict_image_type(image):
    image = np.array(image)
    image = tf.image.resize(image, [224, 224])
    image = np.expand_dims(image, axis=0)
    image = image.squeeze(axis=0)  # 移除多余的维度
    image = image.transpose(1, 2, 0)  # 调整维度顺序为 (height, width, channels)
    prediction = image_model.predict(image)
    type_id = np.argmax(prediction)
    return 饰品类型[type_id]

# 加载推荐系统模型
_, algo = load('recommended.pkl')

# 读取用户-饰品类型评分矩阵
ratings_df = pd.read_csv("C:\\Users\\郑蕊\\Desktop\\饰品推荐系统\\data_transformed.csv")

# 使用推荐系统模型生成推荐列表
def generate_recommendations(user_id, type_id):
    # 使用用户-饰品类型评分矩阵生成推荐列表
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    predictions = [algo.predict(user_id, item_id) for item_id in 饰品图片数据库.keys()]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)
    recommended_types = [r.iid for r in recommendations if r.iid != type_id][:3]
    return recommended_types

# Streamlit应用
st.title('饰品推荐系统')

uploaded_file = st.file_uploader("请上传一张你喜欢的饰品图片", type=["jpg", "png", "jpeg"])

# 如果用户已上传图片
if uploaded_file is not None:
    # 显示上传的图片
    image = PILImage.create(uploaded_file)
    st.image(image,caption='上传的图片',use_column_width=True)
    
    # 获取预测的标签
    type_id, pred_idx, probs = image_model.predict(image)
    st.write(f"预测结果: {type_id}")

    # 显示三张属于该类型的饰品图片
    st.subheader('同类型饰品推荐')
    type_images = 饰品图片数据库[type_id]
    type_images = random.sample(type_images, 3)
    
    if 'type_ratings' not in st.session_state:
        st.session_state.type_ratings = [3] * len(type_images)

    with st.form(key='type_rating_form'):
        for i, img_path in enumerate(type_images):
            img = Image.open(img_path)
            st.image(img, caption=f'{type_id} 饰品', use_column_width=True)
            st.session_state.type_ratings[i] = st.slider('给这张图片打分 (1-5)', 1, 5, st.session_state.type_ratings[i], key=f"type_rating_{i}")
        type_rating_submit_button = st.form_submit_button(label='提交评分')

    if type_rating_submit_button:
        # 显示用户对推荐的同类型饰品图片的满意度
        avg_rating = sum(st.session_state.type_ratings) / len(st.session_state.type_ratings)
        st.write(f'用户对推荐的同类型饰品图片的满意度: {avg_rating:.2f}')

    # 使用推荐系统模型生成推荐列表
    user_id = 200  # 假设用户ID为200
    recommended_types = generate_recommendations(user_id, type_id)

    # 推荐三个用户可能喜欢的其他类型饰品
    st.subheader('其他类型饰品推荐')
    
    if 'other_ratings' not in st.session_state:
        st.session_state.other_ratings = [3] * len(recommended_types)

    with st.form(key='other_rating_form'):
        for i, rec_type in enumerate(recommended_types):
            rec_img_path = random.choice(饰品图片数据库[rec_type])
            rec_img = Image.open(rec_img_path)
            st.image(rec_img, caption=f'{rec_type} 饰品', use_column_width=True)
            st.session_state.other_ratings[i] = st.slider(f'给这张 {rec_type} 饰品图片打分 (1-5)', 1, 5, st.session_state.other_ratings[i], key=f"other_rating_{i}")
        other_rating_submit_button = st.form_submit_button(label='提交评分')

    if other_rating_submit_button:
        # 显示用户对推荐的不同类型饰品的满意度
        avg_other_rating = sum(st.session_state.other_ratings) / len(st.session_state.other_ratings)
        st.write(f'用户对推荐的不同类型饰品的满意度: {avg_other_rating:.2f}')

    # 在最后空一行，然后写提示信息
    st.markdown("")  # 空行
    st.markdown("如果你喜欢我为你推荐的商品，可以保存图片并在某个购物平台上搜索同款哦！")