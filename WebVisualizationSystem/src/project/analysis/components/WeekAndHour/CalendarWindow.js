import React, { useEffect, useState } from 'react';
import './CalendarWindow.scss';
import { Tooltip, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import _ from 'lodash';
// components
import SliderControl from './SliderController';
import WeekHourCalendar from './WeekHourCalendar';
import StatisticsBar from './StatisticsBar';
// react-redux
import { useDispatch, useSelector } from 'react-redux';

 // 天-小时label
 const hoursLabal = (function () {
  let hours = [...Array(24)].map((item, index) => index + 1);
  return hours.map(item => { return `${item}时` })
})()

// 周 label
const weekLabel = ['周日', '周六', '周五', '周四', '周三', '周二', '周一']

const options = [
  {
    type: 'day',
    grid: {
      left: '8%',
      top: '3%',
      right: '2%',
      bottom: '15%',
    },
    xData: hoursLabal
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
    xData: weekLabel
    // acIdx: 3,
  },
]

export default function CalendarWindow(props) {
  const { userData, isVisible, calendarReload, setCalendarReload} = props;
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
      <SliderControl setCalendarReload={setCalendarReload}/>
      <WeekHourCalendar calendarData={calendarData} xLabel={options[0].xData} yLabel={options[1].xData} userData={userData} monthRange={monthRange} calendarReload={calendarReload} />
      <div className='reload-button'>
        <Tooltip title="还原">
          <Button
            ghost
            disabled={false}
            icon={<ReloadOutlined />}
            size={'small'}
            onClick={() => {
              // calendarReload标记，用于后续清除selectedByCalendar数据
              setCalendarReload()
            }}
          />
        </Tooltip>
      </div>
      <StatisticsBar {...options[0]} data={hourCount} isDay={true}/>
      <StatisticsBar {...options[1]} data={weekdayCount} isDay={false}/>
    </div>
  )
}
