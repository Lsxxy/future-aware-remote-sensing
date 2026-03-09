import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import {
    HomeOutlined,
    SearchOutlined,
    PictureOutlined,
    DatabaseOutlined,
    RocketOutlined,
} from '@ant-design/icons';

const { Sider } = Layout; // 使用解构来从Layout中获取Sider组件

const Navigation = () => {
    const location = useLocation();
    const selectedKeys = [location.pathname];
    const [collapsed, setCollapsed] = useState(false);

    return (
        <Sider collapsible collapsed={collapsed} onCollapse={value => setCollapsed(value)}>
            <div className="demo-logo-vertical" style={{ height: 64, width: 200 }} >
                <RocketOutlined style={{ fontSize: '40px', color: '#fff', position: 'relative', left: '60px', top: '13px' }} />
            </div>
            <Menu theme="dark" selectedKeys={selectedKeys} mode="inline" >
                <Menu.Item key="/" icon={<HomeOutlined />} className="menu-item-custom">
                    <NavLink to="/">首页</NavLink>
                </Menu.Item>

                <Menu.Item key="/image-description" icon={<PictureOutlined />}>
                    <NavLink to="/image-description">图像描述</NavLink>
                </Menu.Item>
                
                <Menu.Item key="/MCQ" icon={<SearchOutlined />}>
                    <NavLink to="/MCQ">多项选择题</NavLink>
                </Menu.Item>


            </Menu>
        </Sider>
        // </Layout>
    );
};

export default Navigation;
