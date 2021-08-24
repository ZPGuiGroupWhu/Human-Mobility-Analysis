import React from 'react';
import "./PoiFrame.scss";
import "../css/common.css";
import { Switch, Slider, Input, Radio } from 'antd';
import _ from 'lodash';

const PoiFrame = (props) => {
  const { state, setState, poiInfo, setPoiInfo } = props;
  // POI查询-多选框-选项
  const radioOptions = [
    { label: '起点', value: 'start' },
    { label: '终点', value: 'end' },
  ]
  return (
    <div className="poi-frame-ctn universal-frame-style">
      <div className="item-ctn swtich-ctn">
        <span>{`POI查询`}</span>
        <Switch
          size="middle"
          checkedChildren="开启"
          unCheckedChildren="关闭"
          checked={state}
          onChange={setState}
        />
      </div>
      <div className="item-ctn">
        <div style={{ width: '100px' }}>{`半径(${poiInfo.radius} 米)`}</div>
        <Slider
          min={1}
          max={500}
          defaultValue={poiInfo.radius}
          disabled={!state}
          onChange={
            _.debounce(
              (value) => setPoiInfo({ type: 'radius', payload: value }),
              200,
              { trailing: true }
            )
          }
          tooltipPlacement='left'
          tooltipVisible={false}
          style={{ width: '80px' }}
        />
      </div>
      <div className="item-ctn">
        <span>{`关键词`}</span>
        <Input
          placeholder='POI关键词'
          disabled={!state}
          onChange={
            _.debounce(
              (e) => { setPoiInfo({ type: 'keyword', payload: e.target.value }) },
              500,
              { trailing: true }
            )
          }
          allowClear
          size='middle'
          style={{ width: '100px' }}
        />
      </div>
      <Radio.Group
        name='poiSearch'
        defaultValue={poiInfo.description}
        disabled={!state}
        onChange={(e) => { setPoiInfo({ type: 'description', payload: e.target.value }); }}
      >
        {radioOptions.map((item, idx) => {
          return (
            <Radio value={item.value} key={idx}>{item.label}</Radio>
          )
        })}
      </Radio.Group>
    </div>
  )
}

export default PoiFrame;