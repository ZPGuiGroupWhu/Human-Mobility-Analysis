import React, { useState } from 'react';
import "./PredictFrame.scss";
import { Button } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined } from '@ant-design/icons';

export default function PredictFrame(props) {
  const [activeIdx, setActiveIdx] = useState(-1);

  function setIconColor(idx) {
    return (activeIdx === idx) ? '#fff' : '#111111'
  }
  function setButtonColor(idx) {
    return (activeIdx === idx) ? '#EF5350' : '#fff'
  }

  return (
    <div className="predict-frame-ctn">
      <Button
        onClick={
          () => {
            setActiveIdx(0);
            props.onPredictDispatch({type: 'startPredict'});
          }
        }
        icon={<PlayCircleOutlined style={{ color: setIconColor(0) }} />}
        style={{backgroundColor: setButtonColor(0)}}
        block
      >
        <span style={{color: setIconColor(0)}}>开始预测</span>
      </Button>
      <Button
        onClick={
          () => {
            setActiveIdx(1);
            props.onPredictDispatch({type: 'stopPredict'});
          }
        }
        icon={<PauseCircleOutlined style={{ color: setIconColor(1) }} />}
        style={{backgroundColor: setButtonColor(1)}}
        block
      >
        <span style={{color: setIconColor(1)}}>暂停预测</span>
      </Button>
      <Button
        onClick={
          () => {
            setActiveIdx(2);
            props.onPredictDispatch({type: 'clearPredict'});
          }
        }
        icon={<StopOutlined style={{ color: setIconColor(2) }} />}
        style={{backgroundColor: setButtonColor(2)}}
        block
      >
        <span style={{color: setIconColor(2)}}>结束预测</span>
      </Button>
    </div>
  )
}