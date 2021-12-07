import React from 'react';
import './TimerLine.scss';
import TimerCommonBar from './TimerCommonBar';
import TController from './TController';
// fake data
import jsondata from '@/project/analysis/components/deckGL/399313.json';

function getHourNums(data) {
  let res = new Array(24).fill(0);
  for (let i = 0; i < data.length; i++) {
    res[data[i].hour - 1] += 1
  }
  return res;
}

function getWeekNums(data) {
  let res = new Array(24).fill(0);
  for (let i=0; i<data.length; i++) {
    res[data[i].weekday] += 1
  }
  return res;
}

export default function TimerLine(props) {
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
      data: getHourNums(jsondata),
      acIdx: 7,
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
      data: getWeekNums(jsondata),
      acIdx: 3,
    },
  ]
  return (
    <div className='timer-line-ctn'>
      <TController />
      <TimerCommonBar {...options[0]} />
      <TimerCommonBar {...options[1]} />
    </div>
  )
}
