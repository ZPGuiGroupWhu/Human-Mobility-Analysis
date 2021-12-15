import React from 'react';
import './TimerLine.scss';
import TimerCommonBar from './TimerCommonBar';
import TController from './TController';
import {useSelector} from 'react-redux';

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
    acIdx: 3,
  },
]

export default function TimerLine(props) {
  const {
    getTripsLayer,
    getIconLayer,
  } = props;

  const hourCount = useSelector(state => state.analysis.hourCount);
  const weekdayCount = useSelector(state => state.analysis.weekdayCount);

  return (
    <div className='timer-line-ctn'>
      <TController getTripsLayer={getTripsLayer} getIconLayer={getIconLayer}/>
      <TimerCommonBar {...options[0]} data={hourCount} />
      <TimerCommonBar {...options[1]} data={weekdayCount} />
    </div>
  )
}
