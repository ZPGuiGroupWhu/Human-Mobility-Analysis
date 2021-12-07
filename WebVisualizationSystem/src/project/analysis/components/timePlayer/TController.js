import React from 'react';
import { Slider, Button, Radio } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import './TimerLine.scss';

const map = [{
  0: '一月',
  1: '二月',
  2: '三月',
  3: '四月',
  4: '五月',
  5: '六月',
  6: '七月',
  7: '八月',
  8: '九月',
  9: '十月',
  10: '十一月',
  11: '十二月',
}, {
  0: '周一',
  1: '周二',
  2: '周三',
  3: '周四',
  4: '周五',
  5: '周六',
  6: '周日',
},
getHourMarks()
];

function getHourMarks() {
  let mark = {};
  for (let i = 0; i < 24; i++) {
    mark[i] = `${i + 1}时`
  }
  return mark;
}

export default function TController(props) {
  function showTooltip(map, value) {
    return map[value]
  }

  function onSliderAfterChange(value, idx, callback) {
    console.log(idx, value);
    callback?.(value);
  }

  const options = [
    {min: 0, max: 11},
    {min: 0, max: 6},
    {min: 0, max: 23},
  ]

  return (
    <div className='timer-line-ctrl'>
      {options.map((item, idx) => (
      <Slider 
        dots
        range={{draggableTrack: true}}
        defaultValue={[item.min, item.max]}
        min={item.min}
        max={item.max}
        tipFormatter={value => showTooltip(map[idx], value)}
        onAfterChange={(value) => onSliderAfterChange(value, idx, props.fn)}
      />
      ))}
      <div className='ctrl-btns'>
        <Radio.Group
          onChange={() => console.log(1)}
          defaultValue="a"
          size="small"
          buttonStyle="solid"
          style={{ margin: '10px 0' }}
        >
          <Radio.Button value="day">天</Radio.Button>
          <Radio.Button value="week">周</Radio.Button>
          <Radio.Button value="month">月</Radio.Button>
        </Radio.Group>
        <Button
          type="primary"
          shape='circle'
          icon={<PlayCircleOutlined />}
          onClick={() => console.log(1)}
          className='single-btn'
        />
        <Button
          type="primary"
          shape='circle'
          icon={<PauseCircleOutlined />}
          onClick={() => console.log(1)}
          className='single-btn'
        />
        <Button
          type="primary"
          shape='circle'
          icon={<CloseCircleOutlined />}
          onClick={() => console.log(1)}
          className='single-btn'
        />
      </div>
    </div>
  )
}




