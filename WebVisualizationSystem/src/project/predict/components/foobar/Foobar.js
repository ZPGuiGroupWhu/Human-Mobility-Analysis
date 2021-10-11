import React from 'react';
import "./Foobar.scss";
import PredictFrame from './predict-frame/PredictFrame';
import PoiFrame from './poi-frame/PoiFrame';
import ModelFrame from './model-frame/ModelFrame';

const Foobar = (props) => {
  const {
    // 预测
    onPredictDispatch,
    // poi查询
    poi,
    onPoi,
    poiField,
    setPoiField,
  } = props;

  const options = ['model1', 'model2', 'model3']

  return (
    <div className="foobar-ctn">
      <PoiFrame state={poi} setState={onPoi} poiInfo={poiField} setPoiInfo={setPoiField} />
      <ModelFrame options={options} />
      <PredictFrame onPredictDispatch={onPredictDispatch} />
    </div>
  )
}

export default Foobar;