import React, { useState } from 'react';
import "./PredictFrame.scss";
import '../common.css';
import { Button, Tooltip } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined } from '@ant-design/icons';

/**
 * ！此部分有bug, 问题为：
 * --- 在点击开始预测但并没有点击结束，点击右侧购物车切换轨迹，会出现多条轨迹预测情况。 ---
 * 之后修补该bug时，需要在购物车切换轨迹时，初始化 predictState 和 isStart 属性
 */

export default function PredictFrame(props) {
  const [activeIdx, setActiveIdx] = useState(-1);

  // 设置icon颜色
  function setIconColor() {
    return '#111111'
  }
  // 设置button背景颜色，如果是在预测的时候，点击了暂停，则变为红色
  function setButtonColor(idx) {
    return (activeIdx === 0 && idx === 1) ? '#EF5350' : '#fff'
  }

  const iconSize = { fontSize: '13px' }

  // 预测状态： 开始/暂停，用于开始预测和暂停预测
  const [predictState, setPredictState] = useState(false);

  // 预测是否开始，一旦开始过，就设置为true，用于设置结束预测按钮是否 disabled
  const [isStart, setIsStart] = useState(false);

  return (
    <div className="predict-frame-ctn">
      <div className='predict-button-group'>
        <span className='common-span-style'>功能 </span>
        <Tooltip title={ !predictState ? '开始预测' : '暂停预测'} placement='bottom'>
          <Button
            className='predict-frame-button'
            onClick={
              !predictState ? 
              () => {
                setActiveIdx(0);
                props.onPredictDispatch({ type: 'startPredict' });
                setPredictState(true); // 点击后，设置预测状态为 true，即 开始预测
                setIsStart(true); // 只要点击，就意味着开始预测过
              } 
              :
              () => {
                setActiveIdx(1);
                props.onPredictDispatch({ type: 'stopPredict' });
                setPredictState(false); // 点击后，设置预测状态为 false，即 暂停预测
              }
            }
            size='small'
            icon={ !predictState ? <PlayCircleOutlined style={{ color: setIconColor(), ...iconSize }} /> : <PauseCircleOutlined style={{ color: setIconColor(), ...iconSize }} />}
            style={{ backgroundColor: (!predictState ? setButtonColor(0) : setButtonColor(1))}}
          />
        </Tooltip> 
        <Tooltip title='结束预测' placement='bottom'>
          <Button
            className='predict-frame-button'
            onClick={
              () => {
                setActiveIdx(2);
                props.onPredictDispatch({ type: 'clearPredict' });
                setPredictState(false);
                setIsStart(false); // 清除本次预测，下一次预测设置为还没开始过
              }
            }
            size='small'
            icon={<StopOutlined style={{ color: (isStart ? setIconColor() : 'gray'), ...iconSize }} />}
            style={{ backgroundColor: setButtonColor(2) }}
            disabled={ !isStart }
          >
          </Button>
        </Tooltip>
      </div>
      {props.result ? <span><strong>预测误差: </strong>{`${props.result}米`}</span> : null}
    </div>
  )
}