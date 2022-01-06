import React, { useState } from 'react';
import { Button } from 'antd';
import { SwapOutlined } from '@ant-design/icons';
// 自定义组件
import PagePredict from './PagePredict'; // BMap
import Map3D from './components/deckgl/Map3D'; // DeckGL
// 样式
import './Predict.scss';

const asideWidth = '250px'; // 侧边栏宽度
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
        <section className='predict-reset-container'>剩余部分</section>
        {/* 轨迹3d展示 */}
        <section
          className='predict-map-3d-container'
          style={{
            flex: `0 0 ${asideWidth}`,
          }}
        >
          <Map3D />
        </section>
        <div className='predict-switch-button-container'>
          {/* 侧边栏可视按钮 */}
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
