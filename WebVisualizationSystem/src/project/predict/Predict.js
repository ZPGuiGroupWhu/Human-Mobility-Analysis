import React, { useState, useEffect } from 'react';
import { Button } from 'antd';
import { SwapOutlined } from '@ant-design/icons';
// 自定义组件
import PagePredict from './PagePredict'; // BMap
import Map3D from './components/deckgl/Map3D'; // DeckGL
import EvalChart from './components/eval-chart/EvalChart';
import Modal from './components/modal/Modal'; // 弹窗
// 样式
import './Predict.scss';

export const PredictCtx = React.createContext();
export default function Predict(props) {
  const asideWidth = '250px'; // 侧边栏宽度
  
  const [asideVisible, setAsideVisible] = useState(false); // 侧边栏是否可视
  const [modalVisible, setModalVisible] = useState(false); // 弹窗是否可视

  // 确保 Context Value 对象不可变
  const [ctx, setCtx] = useState({modalVisible, setModalVisible,});
  useEffect(() => {setCtx(prev => ({...prev, modalVisible}))}, [modalVisible])

  return (
    <PredictCtx.Provider value={ctx}>
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
          <section className='predict-reset-container'>
            <EvalChart />
          </section>
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
        {/* 弹窗 */}
        <Modal isVisible={modalVisible} setVisible={setModalVisible} />
      </div>
    </PredictCtx.Provider>
  )
}
