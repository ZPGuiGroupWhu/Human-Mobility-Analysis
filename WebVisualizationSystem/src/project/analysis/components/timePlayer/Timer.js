import React, { useEffect, useState } from 'react';
import { Radio } from 'antd';

let timer = null;
export default function Timer(props) {
  const {
    getSelectData,
    getTripsLayer,
    getIconLayer,
  } = props;

  const clearTimer = (timer) => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      console.log(timer);
    }
  }

  const [info, setInfo] = useState({
    trajs: [],
    OPt: [],
    DPt: [],
  })
  useEffect(() => {
    getTripsLayer(info.trajs);
    getIconLayer(true, info.OPt);
    getIconLayer(false, info.DPt);
  }, [info])

  useEffect(() => {
    return () => {
      clearTimer(timer)
    }
  }, [])

  const [curSelected, setCurSelected] = useState('');

  const onChange = (e) => {
    const value = e.target.value;
    if (curSelected === value) {
      return;
    }
    setCurSelected(value);
    switch (value) {
      case 'day':
        dayTimer(getSelectData);
        break;
      case 'week':
        weekTimer(getSelectData);
        break;
      case 'month':
        monthTimer(getSelectData);
        break;
      default:
        throw new Error('Invalid Value');
    }
  }

  const dayTimer = (fn, wait = 1000) => {
    let startDate = '2018-01-01';
    let endDate = '2018-01-01';

    clearTimer(timer);

    let i = 0;
    let [selectONodes, selectDNodes, selectTrajs] = fn(startDate, endDate);
    console.log(selectONodes, selectDNodes, selectTrajs);
    let lens = selectTrajs.length;
    timer = setInterval(() => {
      setInfo({
        trajs: [selectTrajs[i]],
        OPt: [selectONodes[i]],
        DPt: [selectDNodes[i]]
      })
      i++;
      if (i >= lens) {
        startDate = addDateTime(startDate, 1);
        endDate = addDateTime(endDate, 1);
        console.log(startDate);
        [selectONodes, selectDNodes, selectTrajs] = fn(startDate, endDate);
        lens = selectTrajs.length;
        i = 0;
      }
    }, wait)
  }

  const weekTimer = (fn, wait = 1000) => {
    let startDate = '2018-01-01';
    let endDate = '2018-01-01';

    clearTimer(timer);

    let [selectONodes, selectDNodes, selectTrajs] = fn(startDate, endDate);
    timer = setInterval(() => {
      setInfo({
        trajs: selectTrajs,
        OPt: [selectONodes[0]],
        DPt: [selectDNodes[selectDNodes.length - 1]],
      })
      startDate = addDateTime(startDate, 1);
      endDate = addDateTime(endDate, 1);
      [selectONodes, selectDNodes, selectTrajs] = fn(startDate, endDate);
    }, wait)
  }

  const monthTimer = (fn, wait = 1000) => {
    let startDate = '2018-01-01';
    let endDate = '2018-01-07';

    clearTimer(timer);

    let [selectONodes, selectDNodes, selectTrajs] = fn(startDate, endDate);
    timer = setInterval(() => {
      setInfo({
        trajs: selectTrajs,
        OPt: [selectONodes[0]],
        DPt: [selectDNodes[selectDNodes.length - 1]],
      })
      startDate = addDateTime(startDate, 7);
      endDate = addDateTime(endDate, 7);
      [selectONodes, selectDNodes, selectTrajs] = fn(startDate, endDate);
    }, wait)
  }

  // 日期加 n 天
  const addDateTime = function (val, n = 1) {
    let dateTime = new Date(val);
    dateTime = dateTime.setDate(dateTime.getDate() + n);
    dateTime = new Date(dateTime);
    let y = dateTime.getFullYear();
    let m = dateTime.getMonth() + 1;
    m = m < 10 ? `0${m}` : m;
    let d = dateTime.getDate();
    d = d < 10 ? `0${d}` : d;
    return `${y}-${m}-${d}`;
  }



  return (
    <div>
      <Radio.Group onChange={onChange} defaultValue="a">
        <Radio.Button value="day">天</Radio.Button>
        <Radio.Button value="week">周</Radio.Button>
        <Radio.Button value="month">月</Radio.Button>
      </Radio.Group>
    </div>
  )
}
