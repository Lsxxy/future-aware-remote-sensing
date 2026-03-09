from flask import Flask, request, jsonify, send_file
from PIL import Image
import io # 用于在内存中操作二进制数据
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ... (你的其他 Flask 路由) ...

@app.route('/api/convert-image', methods=['POST'])
def convert_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    try:
        # 使用 Pillow 打开上传的图片文件 (无论它是 TIF, JPG, PNG)
        with Image.open(file.stream) as img:
            # 创建一个内存中的二进制流对象
            png_buffer = io.BytesIO()
            # 将图片以 PNG 格式保存到内存中
            img.save(png_buffer, format="PNG")
            # 获取内存中 PNG 数据的字节
            png_bytes = png_buffer.getvalue()

            # 将 PNG 字节编码成 Base64 字符串
            png_base64 = base64.b64encode(png_bytes).decode('utf-8')

            # 构造一个可以直接在 <img> 标签 src 中使用的 Data URL
            png_data_url = f"data:image/png;base64,{png_base64}"

            # 将这个 Data URL 返回给前端
            return jsonify({"preview_url": png_data_url})

    except Exception as e:

        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5002, debug=False) 