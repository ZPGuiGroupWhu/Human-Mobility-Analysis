import React, { useEffect, useState } from 'react';
import Calendar from './Calendar';
import './BottomCalendar.scss';

export default function BottomCalendar(props) {

  const { userData, timeData, eventName, isVisible, clear} = props;

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
      <Calendar timeData={timeData} userData={userData} eventName={eventName} clear={clear} />
    </div>
  )
}
