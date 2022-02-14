import React, { useEffect, useState } from 'react';
import { Tooltip, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import Calendar from './Calendar';
import './BottomCalendar.scss';

export default function BottomCalendar(props) {

  const { userData, timeData, eventName, isVisible, calendarReload, setCalendarReload} = props;

  // 修改是否可见
  useEffect(() => {
    {
      (isVisible === false) ?
      document.querySelector('.bottom-calendar-ctn').style.display = 'none' :
      document.querySelector('.bottom-calendar-ctn').style.display = 'flex'
    }
  }, [isVisible])

  return (
    <div className="bottom-calendar-ctn">
      <Calendar timeData={timeData} userData={userData} eventName={eventName} calendarReload={calendarReload} />      
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
    </div>
  )
}
