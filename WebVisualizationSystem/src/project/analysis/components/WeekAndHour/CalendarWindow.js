import React, { useEffect, useState } from 'react';
import './CalendarWindow.scss';
import { Tooltip, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import _ from 'lodash';
// components
// import SliderControl from './SliderController';
import WeekHourCalendar from './WeekHourCalendar';
// import StatisticsBar from './StatisticsBar';
// react-redux
import { useDispatch, useSelector } from 'react-redux';
import { setHeatmapData } from '@/app/slice/analysisSlice';
// 网络请求
import { getUserTrajectoryCountBetweenDate } from '@/network';


 // 天-小时label
 const hoursLabal = (function () {
  let hours = [...Array(24)].map((item, index) => index);
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
  const { userData, isVisible, heatmapReload, setHeatmapReload} = props;
  const dispatch = useDispatch()
  // 获取公共数组中的日历数据
  const dateRange = useSelector(state => state.analysis.dateRange);

  // 根据日期，从后台获取统计数据，重新组织数据，并传递
  useEffect(() => {
    const id = '399313';
    const [startDate, endDate] = [...dateRange]
    // 获取calendar数据
    let tempData = []; // 存储过程数据 
    let heatmapData = []; // 存储最终数据
    const getHeatmapData = async () => {
      let getData = await getUserTrajectoryCountBetweenDate(id, startDate, endDate);
      tempData.push(getData['MondayHourCount']);
      tempData.push(getData['TuesdayHourCount']);
      tempData.push(getData['WednesdayHourCount']);
      tempData.push(getData['ThursdayHourCount']);
      tempData.push(getData['FridayHourCount']);
      tempData.push(getData['SaturdayHourCount']);
      tempData.push(getData['SundayHourCount']);
      // 重新组织数据
      for (let i = 0; i < tempData.length; i++) {
        for (let j = 0; j < tempData[i].length; j++) {
            heatmapData.push([i, j, tempData[tempData.length - 1 - i][j]])
        }
    }
    // 重新组织数据
    heatmapData = heatmapData.map(item => {
        return [item[1], item[0], item[2]]
    })
    dispatch(setHeatmapData(heatmapData)); // 更新heatmapData数据
    };
    getHeatmapData();
  }, [dateRange])

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
      <WeekHourCalendar xLabel={options[0].xData} yLabel={options[1].xData} userData={userData} heatmapReload={heatmapReload} />
      <div className='reload-button'>
        <Tooltip title="还原">
          <Button
            ghost
            disabled={false}
            icon={<ReloadOutlined />}
            size={'small'}
            onClick={() => {
              // heatmapReload标记，用于后续清除heatmapSelected数据
              setHeatmapReload()
            }}
          />
        </Tooltip>
      </div>
      {/* 统计图表 */}
      {/* <StatisticsBar {...options[0]} data={hourCount} isDay={true}/>
      <StatisticsBar {...options[1]} data={weekdayCount} isDay={false}/> */}
    </div>
  )
}
