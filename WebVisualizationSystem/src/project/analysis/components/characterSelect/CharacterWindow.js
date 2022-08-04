import React, { useEffect, useState } from 'react';
import ParallelChart from './ParallelChart';
import _ from 'lodash';
// react-redux
import { useSelector } from 'react-redux';

export default function CharacterWindow(props) {
  const { userData, updateParallel} = props; // 用户的轨迹数据

  const state = useSelector(state => state.analysis)

  const [data, setData] = useState([]);

  // 计算平均值
  function getAvg(arr) {
    let sum = 0.0;
    arr.forEach((item) => {
      sum += parseFloat(item);
    })
    return sum / arr.length
  }

  // 处理数据, 根据finalSelected数组中的用户编号, 找出对应的特征数据
  function handleData(data) {
    const trajData = [];
    _.forEach(data, (item, index) => {
      if (state.finalSelected.includes(item.id)) {
        let trajId = item.id; // 轨迹编号
        let totalDis = item.disTotal; // 总距离
        let avgSpeed = getAvg(item.spd); // 平均速度
        let maxAzimuth = Math.max(...item.azimuth); // 最大转向角
        let maxSpd = Math.max(...item.spd) // 最大速度
        let maxDis = Math.max(...item.dis) // 最大距离
        trajData.push([trajId, totalDis, maxDis, avgSpeed, maxSpd, maxAzimuth]);
      }
    });
    return trajData;
  }

  // 传递数据给 parallel进行渲染
  useEffect(() => {
    let data = handleData(userData); // 处理数据
    setData(data); // 更新状态
  }, [updateParallel])

  return (
    <div className='character-window-ctn'>
      <ParallelChart data={data} updateParalel={updateParallel}/>
    </div>
  )
}
