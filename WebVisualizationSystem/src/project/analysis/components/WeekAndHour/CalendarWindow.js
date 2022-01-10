import React, { useEffect, useState } from 'react';
import './CalendarWindow.scss';
import { Button } from 'antd';
import { RedoOutlined } from '@ant-design/icons';
import _ from 'lodash';
// components
import SliderControl from './SliderController';
import WeekHourCalendar from './WeekHourCalendar';
import StatisticsBar from './StatisticsBar';
import { getInitTrajIds } from '../dataHandleFunction/dataHandleFunction';
// react-redux
import { useDispatch, useSelector } from 'react-redux';
import { setCalendarSelected } from '@/app/slice/analysisSlice';

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
  const { userData, isVisible, clear } = props;
  // 获取公共数组中的日历数据、hour统计数据、week统计数据
  const calendarData = useSelector(state => state.analysis.calendarData);
  const hourCount = useSelector(state => state.analysis.hourCount);
  const weekdayCount = useSelector(state => state.analysis.weekdayCount);
  const monthRange = useSelector(state => state.analysis.monthRange);

  // 修改是否可见
  useEffect(() => {
    {
      (isVisible === false) ?
        document.querySelector('.calendar-window-ctn').style.display = 'none' :
        document.querySelector('.calendar-window-ctn').style.display = 'flex'
    }
  }, [isVisible])

  return (
    <div className='calendar-window-ctn'>
      <div className='slider-title'>
        <span style={{ color: '#fff', fontFamily: 'sans-serif', fontSize: '15px', fontWeight: 'bold' }}>{'月份选择'}</span>
      </div>
      <SliderControl />
      <WeekHourCalendar calendarData={calendarData} userData={userData} monthRange={monthRange} clear={clear} />
      <StatisticsBar {...options[0]} data={hourCount} />
      <StatisticsBar {...options[1]} data={weekdayCount} />
    </div>
  )
}
