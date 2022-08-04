import React, { useState, useEffect } from 'react';
import { Skeleton } from 'antd';
// 自定义组件
import PagePredict from './PagePredict'; // BMap
import EvalChart from './components/eval-chart/EvalChart';
import Modal from './components/modal/Modal'; // 弹窗

export const PredictCtx = React.createContext();
export default function Predict(props) {
  const [modalVisible, setModalVisible] = useState(false); // 弹窗是否可视

  // 确保 Context Value 对象不可变
  const [ctx, setCtx] = useState({modalVisible, setModalVisible,});
  useEffect(() => {setCtx(prev => ({...prev, modalVisible}))}, [modalVisible])

  return (
    <PredictCtx.Provider value={ctx}>
      <div style={{height: '100%'}}>
        {/* 主视图 */}
        <PagePredict />
        {/* 弹窗 */}
        <Modal isVisible={modalVisible} setVisible={setModalVisible}>
          {/* <Skeleton active title={{width: '100px'}} paragraph={false} round loading />
          <Skeleton.Image loading />
          <Skeleton active paragraph={{rows: 4}} round loading /> */}
          <EvalChart />
        </Modal>
      </div>
    </PredictCtx.Provider>
  )
}
