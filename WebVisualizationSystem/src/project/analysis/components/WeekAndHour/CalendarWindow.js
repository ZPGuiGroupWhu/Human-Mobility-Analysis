import React, { useEffect, useState } from 'react';
import './CalendarWindow.scss';
import { Button } from 'antd';
import { RedoOutlined } from '@ant-design/icons';
import {useSelector} from 'react-redux';
import SliderControl from './SliderController';
import WeekHourCalendar from './WeekHourCalendar';
import StatisticsBar from './StatisticsBar';

const options = [
  {
    type: 'day',
    grid: {
      left: '8%',
      top: '3%',
      right: '2%',
      bottom: '15%',
    },
    xData: [...(new Array(25)).keys()].slice(1),
    // acIdx: 7,
  },
  {
    type: 'week',
    grid: {
      left: '15%',
    top: '3%',
    right: '2%',
    bottom: '15%',
    },
    xData: [...(new Array(8)).keys()].slice(1),
    // acIdx: 3,
  },
]

export default function CalendarWindow(props) {
  const {userData} = props;
  // 获取公共数组中的日历数据、hour统计数据、week统计数据
  const calendarData = useSelector(state => state.analysis.calendarData);
  const hourCount = useSelector(state => state.analysis.hourCount);
  const weekdayCount = useSelector(state => state.analysis.weekdayCount);

  const [clear, setClear] = useState({});
  const [monthRange, setMonthRange] = useState([]);
  // 获取slider的月份数据
  function getMonthRange(month){
    setMonthRange(month);
  }
  // 是否清空高亮选框
  function getClear(clear){
    setClear(clear);
  }

  return (
    <div className='calendar-window-ctn'>
      <SliderControl getMonthRange={getMonthRange} getClear={getClear}/>
      <WeekHourCalendar calendarData = {calendarData} userData={userData} monthRange={monthRange} clear={clear}/>
      <Button
            ghost
            size='small'
            type='default'
            icon={<RedoOutlined style={{ color: '#fff' }} />}
            onClick={() => {setClear({})}} // 清除筛选
            style={{
              position: 'absolute',
              top: '10px',
              left: '715px',
              zIndex: '99' //至于顶层
            }}
          />
      <StatisticsBar {...options[0]} data={hourCount} />
      <StatisticsBar {...options[1]} data={weekdayCount} />
    </div>
  )
}
