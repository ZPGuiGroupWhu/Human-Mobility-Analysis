import React, { useEffect, useState } from 'react';
import './CharacterWindow.scss';
import ParallelChart from './ParallelChart';
import _ from 'lodash';
import { Button } from 'antd';
import { RedoOutlined } from '@ant-design/icons';
// react-redux
import { useDispatch } from 'react-redux';
import { setCharacterSelected } from '@/app/slice/analysisSlice'

export default function CalendarWindow(props) {
  const { userData, isVisible, clear } = props; // 用户的轨迹数据

  // 修改是否可见
  useEffect(() => {
    {
      (isVisible === false) ?
      document.querySelector('.character-window-ctn').style.display = 'none' :
      document.querySelector('.character-window-ctn').style.display = 'flex'
    }
  }, [isVisible])

  return (
    <div className='character-window-ctn'>
      <ParallelChart data={userData} clear={clear} />
    </div>
  )
}
