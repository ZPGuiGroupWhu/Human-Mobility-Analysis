import React, { useEffect, useRef, useState } from 'react';
import { Slider, Button, Radio } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import './TimerLine.scss';
import { getUserTrajByTime } from '@/network';
import { useDispatch, useSelector } from 'react-redux';
import { setBarData } from '@/app/slice/analysisSlice';

export default function TController(props) {
  const dispatch = useDispatch()

  // 当前操作的进行状态: 正在进行动画播放(true) / 未进行动画播放(false)
  const [curStatus, setCurStatus] = useState(false);

  // 时间范围
  const [hour, setHour] = useState([0, 23]);
  const [weekday, setWeekday] = useState([0, 6]);
  const [month, setMonth] = useState([0, 11]);

  // 滑块回调监听
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

  // 数据请求
  useEffect(() => {
    const fn = async () => {
      try {
        let data = await getUserTrajByTime({
          id: 399313,
          hourMin: hour[0],
          hourMax: hour[1],
          weekdayMin: weekday[0],
          weekdayMax: weekday[1],
          monthMin: month[0],
          monthMax: month[1],
        })
        dispatch(setBarData(data));
      } catch (err) {
        console.log(err);
      }
    }
    fn();
  }, [hour, weekday, month])

  // 更新预测类型
  const [type, setType] = useState('day');
  const typeChange = (e) => { setType(e.target.value) }


  return (
    <div className='timer-line-ctrl'>
      {/* Slider 控件：控制筛选的时间范围 */}
      {options.map((item, idx) => (
        <Slider
          key={idx}
          dots
          range={{ draggableTrack: true }}
          defaultValue={[item.min, item.max]}
          min={item.min}
          max={item.max}
          tipFormatter={value => map[idx][value]}
          onAfterChange={(value) => {
            onSliderAfterChange(value, idx);
          }}
        />
      ))}
      <div className='ctrl-btns'>
        {/* 时间粒度 */}
        <Radio.Group
          onChange={(e) => { typeChange(e) }}
          defaultValue="day"
          size="small"
          buttonStyle="solid"
          style={{ margin: '10px 0' }}
        >
          <Radio.Button value="day">天</Radio.Button>
          <Radio.Button value="week">周</Radio.Button>
          <Radio.Button value="month">月</Radio.Button>
        </Radio.Group>
        {/* 预测按钮 */}
        {
          !curStatus ? (
            <Button
              type="primary"
              shape='circle'
              icon={<PlayCircleOutlined />}
              onClick={() => {
                setCurStatus(true);
              }}
              className='single-btn'
            />
          ) : (
            <Button
              type="primary"
              shape='circle'
              icon={<PauseCircleOutlined />}
              onClick={() => {
                setCurStatus(false);
              }}
              className='single-btn'
            />
          )
        }
        <Button
          type="primary"
          shape='circle'
          icon={<CloseCircleOutlined />}
          onClick={() => {
            setCurStatus(false)
          }}
          className='single-btn'
        />
      </div>
    </div>
  )
}


const options = [
  { min: 0, max: 11 }, // month
  { min: 0, max: 6 }, // weekday
  { min: 0, max: 23 }, // hour
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