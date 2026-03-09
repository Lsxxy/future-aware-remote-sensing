// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import HomePage from './components/HomePage';
import ImageCaption from './components/ImageCaption/ImageUpload';
import MCQ from './components/MCQ/ImageUpload';
import { Layout } from 'antd';
import AaPP from './test';
const { Header, Content, Footer } = Layout;

function App() {
  // return (
  //   <AaPP />
  // )
  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Navigation />
        <Layout style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <Header style={{ background: '#colorBgContainer', padding: 0 }}>Header content here</Header>
          <Content style={{ 
            margin: '0 16px', 
            flex:1,
            display: 'flex',          // 1. 设置为 Flex 布局
            flexDirection: 'column'   // 2. 设置主轴为垂直方向
          }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/image-description" element={<ImageCaption />} />
              <Route path="/mcq" element={<MCQ />} />
            </Routes>
          </Content>
          <Footer style={{ textAlign: 'center' }}>Ant Design ©{new Date().getFullYear()} Created by Ant UED</Footer>
        </Layout>
      </Layout>
    </Router>
  );
}

export default App;


