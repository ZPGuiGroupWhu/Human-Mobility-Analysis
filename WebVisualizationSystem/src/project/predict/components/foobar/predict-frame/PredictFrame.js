import React, { useState } from 'react';
import "./PredictFrame.scss";
import "../css/common.css";
import { Button, Tooltip } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined } from '@ant-design/icons';

export default function PredictFrame(props) {
  const [activeIdx, setActiveIdx] = useState(-1);

  function setIconColor(idx) {
    return (activeIdx === idx) ? '#fff' : '#111111'
  }
  function setButtonColor(idx) {
    return (activeIdx === idx) ? '#EF5350' : '#fff'
  }

  const iconSize = { fontSize: '13px' }

  return (
    <div className="predict-frame-ctn universal-frame-style">
      <div className='predict-button-group'>
        <strong style={{ color: '#fff' }}>功能 </strong>
        <Tooltip title='开始预测'>
          <Button
            className='predict-frame-button'
            onClick={
              () => {
                setActiveIdx(0);
                props.onPredictDispatch({ type: 'startPredict' });
              }
            }
            size='small'
            icon={<PlayCircleOutlined style={{ color: setIconColor(0), ...iconSize }} />}
            style={{ backgroundColor: setButtonColor(0) }}
          >
          </Button>
        </Tooltip>
        <Tooltip title='暂停预测'>
          <Button
            className='predict-frame-button'
            onClick={
              () => {
                setActiveIdx(1);
                props.onPredictDispatch({ type: 'stopPredict' });
              }
            }
            size='small'
            icon={<PauseCircleOutlined style={{ color: setIconColor(1), ...iconSize }} />}
            style={{ backgroundColor: setButtonColor(1) }}
          >
          </Button>
        </Tooltip>
        <Tooltip title='结束预测'>
          <Button
            className='predict-frame-button'
            onClick={
              () => {
                setActiveIdx(2);
                props.onPredictDispatch({ type: 'clearPredict' });
              }
            }
            size='small'
            icon={<StopOutlined style={{ color: setIconColor(2), ...iconSize }} />}
            style={{ backgroundColor: setButtonColor(2) }}
          >
          </Button>
        </Tooltip>
      </div>
      {props.result ? <span style={{ color: '#fff' }}><strong>预测误差: </strong>{`${props.result}米`}</span> : null}
    </div>
  )
}