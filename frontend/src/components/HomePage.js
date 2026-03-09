// 在 src/components 下创建 HomePage.js

import React from 'react';
import { Card, Typography, Button, Row, Col, Space } from 'antd';
import { ScanOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

const { Title, Paragraph, Text } = Typography;

function HomePage() {
    const navigate = useNavigate();

return (
        <div style={{
            width: '100%',
            height: '100%',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '24px'
        }}>
            <Card
                style={{
                    width: '100%',
                    height: '100%', 
                    textAlign: 'center',
                    boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                    display: 'flex',
                    flexDirection: 'column'
                }}
                bodyStyle={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-around', // 让内容均匀分布
                    padding: '24px'
                }}
            >
                {/* --- 1. 标题区 --- */}
                <div>
                    <Title level={2}>欢迎来到遥感图像智能解译系统</Title>
                    <Paragraph type="secondary" style={{ fontSize: '16px' }}>
                        本系统利用先进的多模态大语言模型，为您提供强大的遥感图像理解与分析功能。
                    </Paragraph>
                </div>

                {/* --- 2. 功能区 --- */}
                <Row gutter={[48, 32]} justify="center" align="stretch" style={{ flex: 1, width: '100%', alignItems: 'center' }}>
                    
                    <Col xs={24} sm={12} style={{ display: 'flex' }}>
                        {/* **核心改动 4: 功能卡片高度设为 100%** */}
                        <Card hoverable onClick={() => navigate('/image-description')} style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '16px' }}>
                                <ScanOutlined style={{ fontSize: '4rem', color: '#1890ff' }} />
                                <Title level={4}>图像描述</Title>
                                <Text type="secondary">上传一张遥感图片，模型将自动为其生成详细的文本描述。</Text>
                            </div>
                        </Card>
                    </Col>
                    
                    <Col xs={24} sm={12} style={{ display: 'flex' }}>
                        <Card hoverable onClick={() => navigate('/mcq')} style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '16px' }}>
                                <QuestionCircleOutlined style={{ fontSize: '4rem', color: '#52c41a' }} />
                                <Title level={4}>多项选择题</Title>
                                <Text type="secondary">上传图片并输入问题，让模型根据图像内容回答复杂的多项选择题。</Text>
                            </div>
                        </Card>
                    </Col>

                </Row>

                {/* --- 3. 快速开始区 --- */}
                <div>
                    <Title level={5}>快速开始</Title>
                    <Paragraph>
                        请点击上方功能卡片或左侧导航栏开始您的探索之旅！
                    </Paragraph>
                    <Button type="primary" size="large" onClick={() => navigate('/image-description')}>
                        立即开始图像描述
                    </Button>
                </div>
            </Card>
        </div>
    );
}

export default HomePage;
