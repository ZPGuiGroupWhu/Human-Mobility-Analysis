import React, { useEffect, useState } from 'react';
import "./PoiFrame.scss";
import "../common.css";
import { Switch, Slider, Input, Radio } from 'antd';
import _ from 'lodash';
import FoldPanel from '@/components/fold-panel/FoldPanel';

const PoiFrame = (props) => {
  const { isVisible, state, setState, poiInfo, setPoiInfo } = props;
  // POI查询-多选框-选项
  const radioOptions = [
    { label: '起点', value: 'start' },
    { label: '终点', value: 'end' },
  ]

  const [prevState, setPrevState] = useState(state);

  useEffect(() => {
    if (!isVisible) {
      // 抽屉折叠，则关闭poi检索，同时保留关闭前的poi检索状态
      setPrevState(state);
      setState(false);
    } else {
      // 抽屉展开，恢复之前保留的poi检索状态
      setState(prevState);
    }
  }, [isVisible])

  return (
    <FoldPanel
      width='100%'
      id='poi-frame'
      className='common-margin-bottom'
      renderEntryComponent={(setFold) => (
        <div className="poi-frame-entry">
          <span className='common-span-style'>{`POI查询`}</span>
          <Switch
            size="middle"
            checkedChildren="开启"
            unCheckedChildren="关闭"
            checked={state}
            onChange={() => {
              setState(prev => !prev);
              setFold(prev => !prev);
            }}
          />
        </div>
      )}
      renderExpandComponent={() => (
        <div className='poi-frame-expand'>
          <div className="poi-frame-expand-row">
            <span className='common-span-style' style={{marginRight:'3px'}}>{`半径(${poiInfo.radius} 米)`}</span>
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
          <div className="poi-frame-expand-row">
            <span className='common-span-style' style={{marginRight:'15px'}}>{`关键词`}</span>
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
              style={{ width: '120px' }}
            />
          </div>
          <div className="poi-frame-expand-row">
            <Radio.Group
              name='poiSearch'
              defaultValue={poiInfo.description}
              disabled={!state}
              onChange={(e) => { setPoiInfo({ type: 'description', payload: e.target.value }); }}
            >
              {radioOptions.map((item, idx) => {
                return (
                  <Radio value={item.value} key={idx} style={{color: '#fff'}}>{item.label}</Radio>
                )
              })}
            </Radio.Group>
          </div>
        </div>
      )}
    />
  )
}

export default PoiFrame;