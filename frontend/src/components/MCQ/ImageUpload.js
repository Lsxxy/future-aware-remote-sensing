import React, { useState, useEffect } from 'react';
import { Button, Upload, message, Card, Flex, Spin, Typography, Image, Input, Empty, Tag } from 'antd';
import { InboxOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import './ImageUpload.css';

const { Dragger } = Upload;
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input; // <-- 1. 导入 TextArea 用于多行文本输入

function MCQ() {
    // --- State 管理 ---
    const [previewImage, setPreviewImage] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [questionData, setQuestionData] = useState(null);
    
    // **修改点**: 不再需要 jsonFile state，改为存储 json 文本
    const [jsonText, setJsonText] = useState(''); 

    const [modelOutput, setModelOutput] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    // --- 图片上传处理  ---
    const handleImageUploadChange = async (info) => {
        const { file } = info;
        if (file.status === 'uploading') { setIsLoading(true); return; }
        if (file.status === 'done') {
            setPreviewImage(null);
            setModelOutput(null);
            setImageFile(file.originFileObj);
            const formData = new FormData();
            formData.append('image', file.originFileObj);
            try {
                const response = await fetch('http://localhost:5002/api/convert-image', { method: 'POST', body: formData });
                if (response.ok) {
                    const data = await response.json();
                    setPreviewImage(data.preview_url);
                    message.success(`${file.name} 图片加载成功!`);
                } else {
                    message.error('生成图片预览失败。');
                    setImageFile(null);
                }
            } catch (error) {
                message.error('无法连接到图片预览服务。');
                setImageFile(null);
            } finally {
                setIsLoading(false);
            }
        } else if (file.status === 'error') {
            message.error(`${file.name} 文件上传失败.`);
            setIsLoading(false);
        }
    };

    // --- JSON 文本输入处理 ---
    const handleJsonTextChange = (e) => {
        const text = e.target.value;
        setJsonText(text); // 实时更新输入的文本

        // 简单的防抖：延迟一点时间再解析，避免用户每输入一个字符就解析一次
        const timer = setTimeout(() => {
            if (text.trim() === '') {
                setQuestionData(null); // 如果清空了文本，就清空预览
                return;
            }
            try {
                const json = JSON.parse(text);
                if (json.Text && json['Answer choices']) {
                    setQuestionData(json);
                } else {
                    setQuestionData(null); // 格式不符也清空预览
                }
            } catch (error) {
                // 解析失败时，也清空预览
                setQuestionData(null);
            }
        }, 500); // 延迟500毫秒

        // 清理上一个计时器，实现防抖
        return () => clearTimeout(timer);
    };

    // --- 核心预测函数  ---
    const handlePredict = async () => {
        if (!imageFile || !questionData) {
            message.warning("请上传图片并输入有效的 JSON 问题后再进行预测。");
            return;
        }
        setIsLoading(true);
        setModelOutput(null);

        const formData = new FormData();
        formData.append('image', imageFile);
        
        // 将解析后的 JSON 对象字符串化后发送

        formData.append('json_question', JSON.stringify(questionData));

        try {
            const response = await fetch('http://localhost:5000/predict_mcq', {
                method: 'POST',
                body: formData,
            });
            if (response.ok) {
                const data = await response.json();
                setModelOutput(data.model_answer);
                message.success("预测成功!");
            } else {
                message.error("预测失败，服务器响应错误。");
            }
        } catch (error) {
            console.error('预测过程中出错', error);
            message.error("预测失败，无法连接到预测服务。");
        } finally {
            setIsLoading(false);
        }
    };
    
    const dummyRequest = ({ file, onSuccess }) => {
        setTimeout(() => { onSuccess("ok"); }, 0);
    };

    const uploadButton = (
        <div className="upload-placeholder">
            <p className="ant-upload-drag-icon"><InboxOutlined /></p>
            <p className="ant-upload-text">点击或拖拽图片</p>
            <p className="ant-upload-hint">支持 TIF, PNG, JPG</p>
        </div>
    );

    return (
        <Spin spinning={isLoading} tip="模型正在分析图像，请稍候..." size="large">
            <Flex style={{ padding: '24px', minHeight: 'calc(100vh - 120px)' }} gap="large">

                {/* --- 左侧列 --- */}
                <Flex vertical gap="large" style={{ flex: 1 }}>
                    <Title level={3} style={{ flexShrink: 0 }}>1. 上传与输入</Title>
                    
                    <Card title={
                        <span style={{ fontSize: '22px', fontWeight: 600 }}>
                            上传遥感图片 (TIF, PNG, JPG)
                        </span>
                        } 
                    style={{ flex: 1, display: 'inside' }} bodyStyle={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Upload
                            name="imageFile"
                            listType="picture-card" // <-- 使用 picture-card 类型
                            className="responsive-square-uploader" // <-- 使用新的、统一的 className
                            showUploadList={false}
                            accept=".tif,.tiff,.png,.jpg,.jpeg"
                            customRequest={dummyRequest}
                            onChange={handleImageUploadChange}
                        >
                            {previewImage ? (
                                <img src={previewImage} alt="预览" className="preview-image-fill" />
                            ) : (
                                uploadButton
                            )}
                        </Upload>
                    </Card>
                    
                    <Card title={
                        <span style={{ fontSize: '22px', fontWeight: 600 }}>
                            输入问题内容 (JSON 格式)
                        </span>
                        } >
                         <TextArea
                            value={jsonText}
                            onChange={handleJsonTextChange}
                            placeholder='请在此处粘贴 JSON 文本...'
                            autoSize={{ minRows: 6, maxRows: 7 }}
                            style={{ fontSize: '20px', lineHeight: '1.6' }} 
                        />
                    </Card>
                </Flex>

                {/* --- 右侧列 --- */}
                <Flex vertical gap="large" style={{ flex: 1 }}>
                    <Title level={3} style={{ flexShrink: 0 }}>2. 预览与结果</Title>
                    
                    <Card title={
                        <span style={{ fontSize: '22px', fontWeight: 600 }}>
                            问题预览
                        </span>
                        }  style={{ flex: 1, display: 'flex', flexDirection: 'column' }} bodyStyle={{ flex: 1, overflow: 'hidden' }}>
                        <div className="scrollable-content">
                            {questionData ? (
                                <div>
                                    <Paragraph strong style={{ fontSize: '18px', marginBottom: '8px' }}>问题:</Paragraph>
                                    <Paragraph style={{ fontSize: '18px', marginBottom: '8px' }}>{questionData.Text}</Paragraph>
                                    <Paragraph strong style={{ marginTop: 16, fontSize: '18px' }}>选项:</Paragraph>
                                    {questionData['Answer choices'].map((choice, index) => (
                                        <Paragraph style={{ fontSize: '18px', marginBottom: '8px' }}key={index}>{choice}</Paragraph>
                                    ))}
                                </div>
                            ) : (
                                <Empty description="请在左侧输入有效的 JSON 文本以生成预览" />
                            )}
                        </div>
                    </Card>

                    {/* --- 模型结果框 --- */}
                    <Card title={
                        <span style={{ fontSize: '22px', fontWeight: 600 }}>
                            模型预测结果
                        </span>
                        }
                    style={{ width: 1200, minHeight: 50 }}>
                        {
                            // 使用条件渲染来显示不同内容
                            modelOutput ? (
                                // 1. 如果有预测结果，显示结果
                                <div style={{ textAlign: 'center' }}>
                                    <p style={{ fontSize: '18px' }}>模型选择的答案是:</p>
                                    <Tag color="blue" style={{ fontSize: '32px', padding: '10px 20px', fontWeight: 'bold' }}>
                                        {modelOutput}
                                    </Tag>
                                </div>
                            ) : (
                                // 2. 如果没有结果（初始状态或正在加载），显示提示
                                <Empty description="暂无预测结果" />
                            )
                        }
                    </Card>
                    
                    <Button type="primary" size="large" onClick={handlePredict} disabled={!imageFile || !questionData}>
                        开始预测
                    </Button>
                </Flex>
            </Flex>
        </Spin>
    );
}

export default MCQ;