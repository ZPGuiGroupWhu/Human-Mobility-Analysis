import React, { useEffect, useState } from 'react';
import { Slider, Button, Radio } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import './TimerLine.scss';
import { getUserTrajByTime } from '@/network';
import { useDispatch } from 'react-redux';
import { setAll } from '@/app/slice/analysisSlice';
import axios from 'axios';

const options = [
  { min: 0, max: 11 },
  { min: 0, max: 6 },
  { min: 0, max: 23 },
]

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
  const [hour, setHour] = useState([0, 23]);
  const [weekday, setWeekday] = useState([0, 6]);
  const [month, setMonth] = useState([0, 11]);

  const dispatch = useDispatch()

  async function onSliderAfterChange(value, idx) {
    switch (idx) {
      case 0:
        setMonth(value);
        break;
      case 1:
        setWeekday(value);
        break;
      case 2:
        setHour(value);
        break;
      default:
        break;
    }
  }

  const [cancelList, setCancelList] = useState([]);
  useEffect(() => {
    const fn = async () => {
      if (cancelList.length) {
        cancelList.forEach(item => {item()})
        setCancelList([])
      }
      try {
        let {data} = await axios.get('/getUserTrajByTime', {
          baseURL: 'http://192.168.61.60:8081',
          params: {
            id: 399313,
            hourMin: hour[0],
            hourMax: hour[1],
            weekdayMin: weekday[0],
            weekdayMax: weekday[1],
            monthMin: month[0],
            monthMax: month[1],
          },
          cancelToken: new axios.CancelToken(function executor(c) {
            // executor 函数接收一个 cancel 函数作为参数
            setCancelList(prev => ([...prev, c]));
          }),
        })
        dispatch(setAll(data));
      } catch (err) {
        console.log(err);
      }
    }
    fn();
  }, [hour, weekday, month])

  return (
    <div className='timer-line-ctrl'>
      {options.map((item, idx) => (
        <Slider
          key={idx}
          dots
          range={{ draggableTrack: true }}
          defaultValue={[item.min, item.max]}
          min={item.min}
          max={item.max}
          tipFormatter={value => map[idx][value]}
          onAfterChange={(value) => onSliderAfterChange(value, idx)}
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




