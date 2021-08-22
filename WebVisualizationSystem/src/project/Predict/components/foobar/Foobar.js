import React from 'react';
import "./Foobar.scss";
import PredictFrame from './predict-frame/PredictFrame';

const Foobar = (props) => {
  return (
    <div className="foobar-ctn">
      <PredictFrame onPredictDispatch={props.onPredictDispatch} />
    </div>
  )
}

export default Foobar;