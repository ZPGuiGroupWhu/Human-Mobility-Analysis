import React, { useEffect, useState } from 'react';
import '../foldContent/FoldContent.scss';
import ParallelChart from './ParallelChart';
import _ from 'lodash';
// react-redux
import { useDispatch, useSelector } from 'react-redux';
import { setCharacterSelected } from '@/app/slice/analysisSlice'

export default function CharacterWindow(props) {
  const { userData, characterReload } = props; // 用户的轨迹数据

  const state = useSelector(state => state.analysis)
  const dispatch = useDispatch();

  const [data, setData] = useState([]);
  const [allUsers, setAllUsers] = useState([]);
  const [result, setResult] = useState([]);

  // 被选择的用户编号
  const userId = userData[0].userid;

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
      let avgSpeed = getAvg(item.spd);
      let avgAzimuth = getAvg(item.azimuth);
      let totalDis = item.disTotal;
      trajData.push([totalDis, avgSpeed, avgAzimuth]);
    });
    return trajData;
  }

  // 返回数据
  function returnSelectedResult(value) {
    setResult(value);
  }

  useEffect(() => {
    // 针对api自带的清除工具 如果清空 则返回所有的轨迹编号求交集，反之返回选择的轨迹编号
    (result === '' ? dispatch(setCharacterSelected(allUsers)) : dispatch(setCharacterSelected(result)));
  }, [result])

  // 获取用户 初始的所有轨迹编号
  useEffect(() => {
    let allUsers = [];
    _.forEach(userData, (item, index) => {
      allUsers.push(item.id)
    })
    setAllUsers(allUsers);
  }, [])

  // 传递数据给 parallel进行渲染
  useEffect(() => {
    let data = handleData(userData); // 处理数据
    setData(data); // 更新状态
  }, [])

  return (
      <ParallelChart returnSelectedResult={returnSelectedResult} data={data} characterReload={characterReload} userId={userId} />
  )
}
