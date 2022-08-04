import React from 'react';
import "./Foobar.scss";
import PoiFrame from './poi-frame/PoiFrame';
import HistAndCur from './model-frame/HistAndCur';
import ModelFolder from './model-folder/ModelFolder';
import SYZ from './model-frame/SYZ';


const Model3 = (props) => {
  return (
    <div>3</div>
  )
}


const Foobar = (props) => {
  const {
    isVisible,
    // 预测
    onPredictDispatch,
    // poi查询
    poi,
    onPoi,
    poiField,
    setPoiField,
    // SYZ
    selectedTraj,
    chart,
  } = props;

  const options = [
    {
      name: 'LSI-LSTM',
      url: 'https://sunyunzeng.com/自己第一篇SCI文章-LSI-LSTM-个体出行目的地预测模型/',
      component: <SYZ selectedTraj={selectedTraj} chart={chart} />
    },
    {
      name: '历史-当前',
      component: <HistAndCur />
    },
    {
      name: 'model3',
      component: <Model3 />
    }
  ]

  return (
    <div className="foobar-ctn">
      <PoiFrame isVisible={isVisible} state={poi} setState={onPoi} poiInfo={poiField} setPoiInfo={setPoiField} />
      <ModelFolder options={options} onPredictDispatch={onPredictDispatch} />
    </div>
  )
}

export default Foobar;