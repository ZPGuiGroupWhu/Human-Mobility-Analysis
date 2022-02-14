import React, { useEffect, useReducer } from 'react';
import { Radio } from 'antd';
import eventBus, { SPDAZMACTION } from '@/app/eventBus';

export default function SYZ(props) {
  // 图层可视状态管理
  const initGridLayerVisible = {
    spdShow: false,
    azmShow: false,
  }
  const gridLayerVisibleReducer = (state, action) => {
    switch (action.type) {
      case 'spd':
        return {
          ...initGridLayerVisible,
          spdShow: true,
        }
      case 'azm':
        return {
          ...initGridLayerVisible,
          azmShow: true,
        }
      case 'none':
        return { ...initGridLayerVisible }
      default:
        break;
    }
  }
  const [gridLayerVisible, gridLayerVisibleDispatch] = useReducer(gridLayerVisibleReducer, initGridLayerVisible)
  const onGridLayerChange = (e) => {
    switch (e.target.value) {
      case 'spd':
        gridLayerVisibleDispatch({ type: 'spd' });
        break;
      case 'azm':
        gridLayerVisibleDispatch({ type: 'azm' });
        break;
      case 'none':
        gridLayerVisibleDispatch({ type: 'none' });
        break;
      default:
        break;
    }
  }

  // 组件销毁前，取消当前组件操作所有产生的结果
  useEffect(() => {
    return () => {
      gridLayerVisibleDispatch({type: 'none'});
      eventBus.emit(SPDAZMACTION, gridLayerVisible)
    }
  }, [])

  useEffect(() => {
    eventBus.emit(SPDAZMACTION, gridLayerVisible)
  }, [gridLayerVisible])

  return (
    <>
      <Radio.Group size={"small"} style={{ width: '100%', display: 'flex' }} buttonStyle="solid" onChange={onGridLayerChange} defaultValue="none">
        <Radio.Button style={{ width: '100%', textAlign: 'center' }} value="spd">速度</Radio.Button>
        <Radio.Button style={{ width: '100%', textAlign: 'center' }} value="azm" >转向角</Radio.Button>
        <Radio.Button style={{ width: '100%', textAlign: 'center' }} value="none">关闭</Radio.Button>
      </Radio.Group>
    </>
  )
}
