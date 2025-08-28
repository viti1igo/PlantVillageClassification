import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. 加载模型 ---
# 确保 'plant_disease_classifier.h5' 文件和 app.py 在同一个目录下，
# 或者提供正确的相对/绝对路径。
MODEL_PATH = './model_classificayion/Model_CNN2.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"模型从 {MODEL_PATH} 加载成功。")
except Exception as e:
    print(f"错误：无法加载模型 {MODEL_PATH}。请确保文件存在且与 app.py 在同一目录，或者路径正确。")
    print(f"详细错误: {e}")

    model = None # 或者 raise SystemExit("无法加载模型")

# --- 2. 定义类别名称 ---
# !!! 关键：确保这个列表的顺序与你训练时的 class_indices 完全一致 !!!
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Common_rust', 'Gray_leaf_spot', 'Northern_Leaf_Blight', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
    'healthy'
]
print(f"已定义的类别数量: {len(CLASS_NAMES)}")

# --- 3. 定义预处理函数 ---
# 输入形状应与模型定义中的 input_shape 匹配 (224, 224, 3)
INPUT_SHAPE = (224, 224)

def preprocess_image(img_pil):
    """将 PIL 图像转换为模型所需的格式"""
    img_pil = img_pil.resize(INPUT_SHAPE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_pil)
    # 标准化/归一化 (如果训练时使用了，这里也要用。例如除以 255)
    # 如果你的 ImageDataGenerator 使用了 rescale=1./255，这里也要做同样的操作
    img_array = img_array / 255.0 # 假设训练时做了归一化到 [0,1]
    # 添加批次维度
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- 4. 定义预测函数 ---
def predict(img_pil):

    if model is None:
        return {"错误": "模型未能加载，无法进行预测。"}
    if not isinstance(img_pil, Image.Image):
        return {"错误": "输入无效，需要 PIL.Image 对象。"}

    try:
        # 预处理图像
        processed_image = preprocess_image(img_pil)

        # 模型预测
        predictions = model.predict(processed_image)[0] # 获取第一个（也是唯一一个）结果

        # 检查输出维度是否与类别数量匹配
        if len(predictions) != len(CLASS_NAMES):
             print(f"警告：模型输出维度 ({len(predictions)}) 与类别名称数量 ({len(CLASS_NAMES)}) 不匹配！")
             # 可以尝试截断或填充，但这通常表示类别列表或模型有问题
             # return {"错误": "模型输出维度与定义的类别数量不匹配。"}

        # 将预测概率与类别名称配对
        confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

        return confidences

    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return {"错误": f"预测失败: {e}"}

# --- 5. 创建 Gradio 界面 ---
# 添加一些示例图片 (确保这些图片文件也上传到你的 Space 仓库)
example_files = ["example1.jpeg", "example2.jpeg"] # 替换成你实际的示例图片文件名
examples_exist = [f for f in example_files if os.path.exists(f)]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a picture of a plant leaf"),
    outputs=gr.Label(label="Probability of disease classification", num_top_classes=5), # 显示概率最高的 5 个类别
    title="🌿 Plant Disease Classifier",
    description="Upload a picture of a plant leaf and the model will predict the category of disease it may have and its probability. The model is built on TensorFlow/Keras.",
    examples=examples_exist if examples_exist else None, # 只有当示例文件存在时才显示
    cache_examples=True if examples_exist else False, # 如果有示例，可以缓存结果加快加载
    allow_flagging="never" # 可以根据需要调整是否允许用户标记结果
)

# --- 6. 启动应用 ---
if __name__ == "__main__":
    # 当在 Hugging Face Spaces 上运行时，它会自动处理启动
    # 本地运行时，可以通过 python app.py 启动
    demo.launch()