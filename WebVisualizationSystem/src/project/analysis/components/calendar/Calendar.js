import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { debounce } from '@/common/func/debounce';
import { eventEmitter } from '@/common/func/EventEmitter';
import { useSelector, useDispatch } from 'react-redux';
import { setSelectedTraj } from '@/app/slice/predictSlice';
import { setTimeSelectResult } from '@/app/slice/analysisSlice';

import _ from 'lodash';


export default function Calendar(props) {
  const myChart = useRef(null);
  const {
    timeData, // 数据(年) - {'yyyy-MM-dd': {count: 2, ...}, ...}
    userData, // 轨迹数据
  } = props;
  const year = str2date(Object.keys(timeData)[0]).getFullYear(); // 数据年份
  const originDate = {
    start: '2018-01-01',
    end: '2018-12-31',
  };

  // 获取筛选的轨迹数据
  const state = useSelector(state => state.predict);
  const dispatch = useDispatch();
  // ECharts 容器实例
  const ref = useRef(null);

  // 根据筛选的起始日期与终止日期，高亮数据
  function highLightData(obj, startDate, endDate) {
    let start = +echarts.number.parseDate(startDate);
    let end = +echarts.number.parseDate(endDate);
    let dayTime = 3600 * 24 * 1000;
    let data = [];
    for (let time = start; time <= end; time += dayTime) {
      const date = echarts.format.formatTime('yyyy-MM-dd', time);
      data.push({
        value: [date, Reflect.get(obj, date)?.count || 0],
        symbol: 'rect',
        itemStyle: {
          borderColor: '#81D0F1',
          borderWidth: 2,
          borderType: 'solid'
        }
      });
    }
    return data;
  }

  // 高亮所选择的轨迹对应的日期格, selectedTraj为选择的轨迹
  function highlightSelectedTrajectoryDate(obj, selectedTraj) {
    const selectedDate = selectedTraj.date;
    let data = [];
    data.push({
      value: [selectedDate, Reflect.get(obj, selectedDate)?.count || 0],
      symbol: 'rect',
      itemStyle: {
        color: "rgba(0, 0, 0, 0)",
        borderColor: 'red',
        borderWidth: 2,
        borderType: 'solid'
      }
    })
    return data;
  }
  // 基于网页宽度，动态计算cell的宽度
  const clientWidth = document.body.clientWidth;
  const cellWidth = (clientWidth - 310) / 55;
  // drawer高度170，减去top padding 20, bottom padding 10, 月份数字高度10，
  const cellHeight = (170 - 20 - 10 - 10) / 7;
  const cellSize = [cellWidth, cellHeight]; // 日历单元格大小
  const option = {
    // title: {
    //   top: 10,
    //   left: 'center',
    //   text: year + '年用户出行统计',
    // },
    tooltip: {
      formatter: function (params) {
        return '日期: ' + params.value[0] + '<br />' + '轨迹数目: ' + params.value[1];
      }
    },
    visualMap: {
      min: 0,
      max: 100,
      calculable: true,
      orient: 'vertical',
      left: 'right',
      top: 10,
      itemHeight: 130,
      textStyle: {
        color: '#fff',
      },
      precision: 0,
      align: 'bottom',
      formatter: function (value) {
        return parseInt(value)
      }
    },
    calendar: {
      top: 20,
      bottom: 10,
      left: 30,
      right: 80,
      cellSize: cellSize,
      range: year || +new Date().getFullYear(), // 日历图坐标范围(某一年)
      itemStyle: {
        borderWidth: 0.5
      },
      dayLabel: {
        color: '#fff',
        nameMap: 'cn',
      },
      monthLabel: {
        color: '#fff',
        nameMap: 'cn',
      },
      yearLabel: { show: false }
    },
    series: [{
      type: 'heatmap',
      coordinateSystem: 'calendar',
      data: [],
      zlevel: 0,
    }, {
      type: 'scatter',
      name: '高亮',
      coordinateSystem: 'calendar',
      symbolSize: cellSize,
      data: [],
      zlevel: 1,
    }, { //增加选中图层
      type: 'scatter',
      name: 'select',
      coordinateSystem: 'calendar',
      symbolSize: cellSize,
      data: [],
      zlevel: 2,
    }]
  }

  // 初始化 ECharts 实例对象
  useEffect(() => {
    if (!ref.current) return () => { };
    myChart.current = echarts.init(ref.current);
    myChart.current.setOption(option);
  }, [ref])


  function addZero(number) {
    if (number < 10) {
      return '0' + number
    } else {
      return number
    }
  }

  function date2str(date, sperator) {
    const year = date.getFullYear();
    const month = addZero(date.getMonth() + 1);
    const day = addZero(date.getDate());
    return year + sperator + month + sperator + day;
  }

  // strDate: yyyy-MM-dd
  function str2date(strDate) {
    strDate.replace('-', '/');
    return new Date(strDate);
  }

  function formatData(obj) {
    // const year = str2date(Object.keys(obj)[0]).getFullYear();
    let start = +echarts.number.parseDate(year + '-01-01');
    let end = +echarts.number.parseDate((+year + 1) + '-01-01');
    let dayTime = 3600 * 24 * 1000;
    let data = [];
    for (let time = start; time < end; time += dayTime) {
      const date = echarts.format.formatTime('yyyy-MM-dd', time)
      data.push([
        date,
        Reflect.get(obj, date)?.count || 0 // 没有数据用 0 填充
      ]);
    }
    return data;
  }

  useEffect(() => {
    if (!timeData) return () => { };
    const format = formatData(timeData);
    const counts = format.map(item => (item[1]))
    myChart.current.setOption({
      visualMap: {
        min: Math.min(...counts),
        max: Math.max(...counts)
      },
      series: {
        data: format,
      }
    })
  }, [timeData])

  function dataFormat(traj) {
    let path = []; // 组织为经纬度数组
    let importance = []; // 存储对应轨迹每个位置的重要程度
    for (let j = 0; j < traj.lngs.length; j++) {
      path.push([traj.lngs[j], traj.lats[j]]);
      // 计算重要程度，判断speed是否为0
      importance.push(
        traj.spd[j] === 0 ?
          traj.azimuth[j] * traj.dis[j] / 0.00001 :
          traj.azimuth[j] * traj.dis[j] / traj.spd[j]);
    }
    // 组织数据, 包括id、date(用于后续选择轨迹时在calendar上标记)、data(轨迹）、spd（轨迹点速度）、azimuth（轨迹点转向角）、importance（轨迹点重要程度）
    let res = {
      id: traj.id,
      date: traj.date,
      data: path,
      spd: traj.spd,
      azimuth: traj.azimuth,
      importance: importance,
      // 新添加了细粒度时间特征
      weekday: traj.weekday + 1,
      hour: traj.hour,
    };
    return res;
  }
  function getSelectDataByDate(start, end) {
    let selectTrajs = [];
    let startTimeStamp = Date.parse(start);
    let endTimeStamp = Date.parse(end);
    for (let i = 0; i < userData.length; i++) {
      if (startTimeStamp <= Date.parse(userData[i].date) && Date.parse(userData[i].date) <= endTimeStamp) {
        selectTrajs.push(dataFormat(userData[i]));
      }
    }
    return selectTrajs  //返回选择的轨迹信息 (OD信息直接读取Trajs的首尾坐标)
  }

  // 记录框选的日期范围
  const [date, setDate] = useState({ start: '', end: '' });
  // 记录鼠标状态
  const [action, setAction] = useState(() => ({ mousedown: false, mousemove: false }));
  // 确保函数只执行一次
  const isdown = useRef(false);
  useEffect(() => {
    const wait = 100;

    if (!myChart.current) return () => { };
    // 鼠标按下事件
    myChart.current.on('mousedown', (params) => {
      dispatch(setSelectedTraj({}));
      if (isdown.current) return;
      // console.log(params.data);
      // 已触发，添加标记
      isdown.current = true;
      // params.data : (string | number)[] such as ['yyyy-MM-dd', 20]
      setAction(prev => {
        return {
          ...prev,
          mousedown: true,
        }
      })
      setDate({
        start: params.data[0] || params.data.value[0],
        end: params.data[0] || params.data.value[0],
      })
    })

    // 鼠标移动事件
    const selectDate = debounce(
      (params) => {
        if (date.end === params.data[0]) return;
        // 记录鼠标状态
        setAction(prev => (
          {
            ...prev,
            mousemove: true,
          }
        ))
        setDate(prev => (
          {
            ...prev,
            end: params.data[0] || params.data.value[0],
          }
        ))
      },
      wait,
      false
    )
    const mouseMove = (params) => {
      action.mousedown && selectDate(params);
    }
    myChart.current.on('mousemove', mouseMove);

    // 鼠标抬起事件
    const mouseUp = (params) => {
      // 重置鼠标状态
      setAction(() => ({ mousedown: false, mousemove: false }));
      // 清除标记
      isdown.current = false;

      let start = date.start, end = date.end;
      let startDate = str2date(start), endDate = str2date(end);
      // 校正时间顺序
      (
        (startDate.getMonth() > endDate.getMonth()) ||
        (startDate.getDay() > endDate.getDay())
      ) && ([start, end] = [end, start])

      // 触发 eventEmitter 中的注册事件，传递选择的日期范围
      // start: yyyy-MM-dd
      // end: yyyy-MM-dd
      const timeSelectedReuslt = getSelectDataByDate(start, end);
      dispatch(setTimeSelectResult(timeSelectedReuslt));
      console.log(start, end);
    }
    myChart.current.on('mouseup', mouseUp)

    return () => {
      myChart.current.off('mousemove', mouseMove);
      myChart.current.off('mouseup', mouseUp);
    }
  }, [myChart.current, date, action])


  // 高亮筛选部分
  useEffect(() => {
    if (!date.start || !date.end) return () => { }
    myChart.current?.setOption({
      series: [{
        name: '高亮',
        data: highLightData(timeData, date.start, date.end),
      }]
    })
  }, [timeData, date])

  // 筛选轨迹的日期标记部分
  useEffect(() => {
    if (!timeData) return () => { }
    myChart.current?.setOption({
      series: [{
        name: 'select',
        data: highlightSelectedTrajectoryDate(timeData, state.selectedTraj)
      }]
    })
  }, [timeData, state.selectedTraj])


  // 清除高亮和标记
  useEffect(() => {
    // 清除高亮
    setTimeout(() => {
      myChart.current?.setOption({
        series: [{
          name: '高亮',
          data: [],
        },{
          name: 'select',
          data: []
        }]
      });
    }, 0);
    // 初始轨迹数据
    let originTrajs = getSelectDataByDate(originDate.start, originDate.end);
    dispatch(setTimeSelectResult(originTrajs));
  }, [props.clear])

  return (
    <div
      ref={ref}
      style={{
        width: '100%',
        height: '100%',
      }}
    ></div>
  )
}
