import React, { useEffect, useReducer, useState } from 'react';
import { Radio } from 'antd';
import eventBus, { SPDAZMACTION } from '@/app/eventBus';
import RelationChart from '@/project/predict/components/charts/relation-chart/RelationChart.js'; // EChart关系折线图

export default function SYZ(props) {
  const { chart, selectedTraj } = props;
  const styleForRadioButton = { width: '100%', textAlign: 'center', fontSize: '10px', fontWeight: 'bold' } // Radio-Button样式

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
      gridLayerVisibleDispatch({ type: 'none' });
      eventBus.emit(SPDAZMACTION, gridLayerVisible)
    }
  }, [])

  useEffect(() => {
    eventBus.emit(SPDAZMACTION, gridLayerVisible)
  }, [gridLayerVisible])

  // 统计图表-地图 联动高亮
  const [highlightData, setHighlightData] = useState([]);
  function onHighlight(idx) {
    setHighlightData((idx >= 0) ? [selectedTraj.data[idx]] : []);
  }
  useEffect(() => {
    if (!chart) return () => { };
    chart.setOption({
      series: [{
        name: '高亮点',
        data: highlightData,
      }]
    });
  }, [chart, highlightData])

  return (
    <>
      <Radio.Group
        size={"small"}
        style={{ width: '100%', display: 'flex', padding: '5px', backgroundColor: '#fff', borderRadius: '5px' }}
        buttonStyle="solid"
        onChange={onGridLayerChange}
        defaultValue="none"
      >
        <Radio.Button style={styleForRadioButton} value="spd">速度</Radio.Button>
        <Radio.Button style={styleForRadioButton} value="azm" >转向角</Radio.Button>
        <Radio.Button style={styleForRadioButton} value="none">关闭</Radio.Button>
      </Radio.Group>
      {/* 速度/转向角关系图 */}
      <RelationChart
        titleText='时间 - 速度/转向角'
        legendData={['速度', '转向角']}
        xAxisData={Array.from({ length: selectedTraj?.spd?.length })}
        yAxis={['速度(km/h)', '转向角(rad)']}
        data={[selectedTraj?.spd, selectedTraj?.azimuth]}
        onHighlight={onHighlight}
      />
    </>
  )
}
