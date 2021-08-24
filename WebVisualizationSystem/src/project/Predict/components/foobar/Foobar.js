import React from 'react';
import "./Foobar.scss";
import PredictFrame from './predict-frame/PredictFrame';
import PoiFrame from './poi-frame/PoiFrame';

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
  return (
    <div className="foobar-ctn">
      <PredictFrame onPredictDispatch={onPredictDispatch} />
      <PoiFrame state={poi} setState={onPoi} poiInfo={poiField} setPoiInfo={setPoiField} />
    </div>
  )
}

export default Foobar;