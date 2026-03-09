import React, { useState } from 'react';
import { LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { Spin, Flex, Card, Upload, Button, Empty, message,Typography } from 'antd';
import './ImageUpload.css';

const { Meta } = Card;
const { Title, Text, Paragraph } = Typography;

function ImageCaption() {
    // // --- State 管理 ---
    const [previewImage, setPreviewImage] = useState(null); // 只用于存储预览的PNG
    const [originalFile, setOriginalFile] = useState(null); // 存储原始上传的文件，用于预测
    const [predictionResults, setPredictionResults] = useState({ /* ... */ });
    const [loading, setLoading] = useState(false);



    // 创建一个自定义的请求处理函数
    const dummyRequest = ({ file, onSuccess }) => {
        setTimeout(() => {
            onSuccess("ok");
        }, 0);
    };


    // 当用户选择图片后触发的函数
    const handleChange = async (info) => {
        if (info.file.status === 'uploading') {
            setLoading(true);
            return;
        }

        // 使用 info.file.originFileObj 获取原始文件
        const file = info.file.originFileObj;
        if (!file) {
            setLoading(false);
            return;
        }

        // 清空旧数据
        setPreviewImage(null);
        setPredictionResults({ /* ... */ });
        setOriginalFile(file); // 存储原始文件，用于稍后的预测

        const formData = new FormData();
        formData.append('image', file);

        try {
            // **调用后端转换 API**
            const response = await fetch('http://localhost:5002/api/convert-image', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                setPreviewImage(data.preview_url); // 用后端返回的 PNG 来设置预览
                message.success('图片预览生成成功!');
            } else {
                message.error('生成图片预览失败。');
            }
        } catch (error) {
            message.error('无法连接到预览服务。');
        } finally {
            setLoading(false); // 无论成功失败，都结束加载动画
        }
    };


    // --- 核心预测函数 ---
    const handlePredict = async () => {
            // 1. 判断条件不变，依然是检查用户是否已上传文件
            if (!originalFile) {
                message.warning("请先上传一张图片再进行预测");
                return;
            }
            setLoading(true);
            // 清空上一次的结果，避免新旧内容混淆**
            setPredictionResults({}); 

            try {
                // 2. 创建一个新的 FormData 对象
                const formData = new FormData();

                // 3. 将【原始文件对象】追加到 FormData 中
                formData.append('image', originalFile);

                // 4. 发送 FormData 到你的【预测模型后端】
                const response = await fetch('http://localhost:5000/predict_caption', {
                    method: 'POST',
                    body: formData,
                });

                // 5. 处理响应
                if (response.ok) {
                    const data = await response.json();
                    setPredictionResults({
                        description: data.description,
                    });
                    message.success("预测成功!");
                } else {
                    // 如果后端返回错误，可以尝试读取错误信息
                    const errorData = await response.json().catch(() => ({ message: "无法解析错误信息" }));
                    console.error('服务器响应错误:', errorData);
                    message.error(`预测失败: ${errorData.message || '服务器响应错误'}`);
                }
            } catch (error) {
                console.error('预测过程中出错', error);
                message.error("预测失败，无法连接到预测服务。");
            } finally {
            setLoading(false);
        }
        };


    // --- UI 渲染部分 ---
    const uploadButton = (
        <button style={{ border: 0, background: 'none' }} type="button">
            {loading ? <LoadingOutlined /> : <PlusOutlined />}
            <div style={{ marginTop: 8 }}>Upload</div>
        </button>
    );

return (
        <Spin spinning={loading} tip="模型正在生成描述中..." size="large">
            <Flex style={{ padding: '24px', minHeight: 'calc(100vh - 120px)' }} gap="large">

                {/* --- 左侧列：上传 --- */}
                <Flex vertical gap="large" style={{ flex: 1 }}>
                    <Title level={3} style={{ flexShrink: 0 }}>1. 上传图片</Title>

                    <Card title={
                        <span style={{ fontSize: '22px', fontWeight: 600 }}>
                            上传遥感图片 (TIF, PNG, JPG)
                        </span>
                        }
                    style={{ flex: 1, display: 'inside', flexDirection: 'column' }} bodyStyle={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Upload
                            name="imageFile"
                            listType="picture-card"
                            className="avatar-uploader"
                            showUploadList={false}
                            accept=".tif,.tiff,.png,.jpg,.jpeg"
                            customRequest={dummyRequest}
                            onChange={handleChange}
                        >
                            {previewImage ? (
                                <img src={previewImage} alt="预览" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                            ) : (
                                uploadButton
                            )}
                        </Upload>
                    </Card>

                    <Button 
                        type="primary" 
                        size="large" 
                        onClick={handlePredict} 
                        disabled={!originalFile || loading}
                        loading={loading}
                    >
                        生成图像描述
                    </Button>
                </Flex>

                {/* --- 右侧列：结果 --- */}
                <Flex vertical gap="large" style={{ flex: 1 }}>
                    <Title level={3} style={{ flexShrink: 0 }}>2. 生成结果</Title>
                    <Card title={
                        <span style={{ fontSize: '22px', fontWeight: 600 }}>
                            模型生成的图像描述
                        </span>
                        }
                        style={{ flex: 1, display: 'flex', flexDirection: 'column' }} 
                        bodyStyle={{ 
                            flex: 1, 
                            display: 'flex', 
                            justifyContent: 'center', 
                            alignItems: 'center' 
                        }}
                    >
                        {loading ? (
                            // 1. 如果正在加载，显示 Spin 组件
                            <Spin tip="模型正在分析图像，请稍候..." size="large" />
                        ) : predictionResults.description ? (
                            // 2. 如果加载完成且有结果，显示结果
                            <div style={{ width: '100%', height: '100%', overflowY: 'auto', padding: '16px' }}>
                                <p style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: '20px' }}>
                                    {predictionResults.description}
                                </p>
                            </div>
                        ) : (
                            // 3. 如果不加载也没结果（初始状态），显示 Empty
                            <Empty description="请先在左侧上传图片并点击生成按钮" />
                        )}
                    </Card>
                </Flex>
            </Flex>
        </Spin>
    );
}

export default ImageCaption;