import React from 'react';
import './CalendarWindow.scss';
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
  // 获取公共数组中的日历数据、hour统计数据、week统计数据
  const calendarData = useSelector(state => state.analysis.calendarData);
  const hourCount = useSelector(state => state.analysis.hourCount);
  const weekdayCount = useSelector(state => state.analysis.weekdayCount);

  return (
    <div className='calendar-window-ctn'>
      <SliderControl />
      <WeekHourCalendar data = {calendarData}/>
      <StatisticsBar {...options[0]} data={hourCount} />
      <StatisticsBar {...options[1]} data={weekdayCount} />
    </div>
  )
}
