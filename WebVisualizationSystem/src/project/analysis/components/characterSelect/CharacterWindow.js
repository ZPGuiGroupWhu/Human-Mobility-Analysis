import React, { useEffect, useState } from 'react';
import './CharacterWindow.scss';
import ParallelChart from './ParallelChart';


export default function CalendarWindow(props) {
  const {userData} = props; // 用户的轨迹数据

  return (
    <div className='character-window-ctn'>
      <ParallelChart data = {userData}/>
    </div>
  )
}
