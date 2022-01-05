import React, { useState } from 'react';
import { Button } from 'antd';
import { SwapOutlined } from '@ant-design/icons';
import PagePredict from './PagePredict';
// 样式
import './Predict.scss';

const asideWidth = '350px'; // 侧边栏宽度
export default function Predict(props) {
  const [asideVisible, setAsideVisible] = useState(false); // 侧边栏是否可视

  return (
    <div className='predict-layout' style={{ width: !asideVisible ? `calc(100% + ${asideWidth})` : '100%' }}>
      {/* 主视图 */}
      <div className='predict-main-content'>
        <PagePredict />
      </div>
      {/* 侧边栏 */}
      <aside
        className='predict-minor-content'
        style={{
          flex: `0 0 ${asideWidth}`,
        }}
      >
        <div className='predict-switch-button-container'>
          <Button
            size='middle'
            type='primary'
            shape='circle'
            icon={<SwapOutlined />}
            onClick={(e) => {
              setAsideVisible(prev => !prev);
            }}
          ></Button>
        </div>
      </aside>
    </div>
  )
}
