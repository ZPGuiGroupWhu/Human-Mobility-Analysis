import React, { useState, useEffect, useRef, useContext } from 'react';
import * as echarts from 'echarts';
import { debounce } from '@/common/func/debounce';
import { eventEmitter } from '@/common/func/EventEmitter';
import _, { forEach } from 'lodash';
// react-redux
import { useSelector, useDispatch } from 'react-redux';
import { setSelectedByCalendar } from '@/app/slice/selectSlice';

let myChart = null;
let timePeriod = [];//存储需要高亮的时间段
const dateFlag = {} //日期是否可选取的标记flag

export default function Calendar(props) {
  const {
    data, // 数据(年) - {'yyyy-MM-dd': {count: 2, ...}, ...}
    bottomHeight,
    bottomWidth,
    calendarReload,
  } = props;

  const state = useSelector(state => state.select);
  const dispatch = useDispatch();

  const year = Object.values(data).length ? str2date(Object.keys(data)[0]).getFullYear() : 2019;  // 数据年份

  // 初始化每个日期的flag
  function initDateFlag() {
    let start = +echarts.number.parseDate(year + '-01-01');
    let end = +echarts.number.parseDate((+year + 1) + '-01-01');
    let dayTime = 3600 * 24 * 1000;
    for (let time = start; time < end; time += dayTime) {
      const date = echarts.format.formatTime('yyyy-MM-dd', time);
      dateFlag[date] = true;
    }
  }

  // ECharts 容器实例
  const ref = useRef(null);

  // 根据筛选的起始日期与终止日期，高亮数据
  function highLightData(obj, startDate, endDate) {
    //存储需要高亮的轨迹日期及其数据
    let data = [];
    //判断向data中添加历史时间段只执行一次
    let addFinish = false;
    //添加历史日期
    for (let i = 0; i < timePeriod.length; i++) {
      //如果没有添加结束则可以继续添加
      if (addFinish === false) {
        let start = +echarts.number.parseDate(timePeriod[i].start);
        let end = +echarts.number.parseDate(timePeriod[i].end);
        let dayTime = 3600 * 24 * 1000;
        for (let time = start; time <= end; time += dayTime) {
          const date = echarts.format.formatTime('yyyy-MM-dd', time);
          data.push({
            value: [date, Reflect.get(obj, date)?.count || 0],
            symbol: 'rect',
            itemStyle: {
              borderColor: '#00BFFF',
              borderWidth: 1,
              borderType: 'solid'
            }
          });
        }
      }
      //如果已经添加到最后一个时间段，则将addFinish标记为true,并跳出
      if (i === timePeriod.length - 1) {
        addFinish = true;
        break
      }
    }
    //添加当前筛选的日期数据
    let start = +echarts.number.parseDate(startDate);
    let end = +echarts.number.parseDate(endDate);
    let dayTime = 3600 * 24 * 1000;
    for (let time = start; time <= end; time += dayTime) {
      const date = echarts.format.formatTime('yyyy-MM-dd', time);
      data.push({
        value: [date, Reflect.get(obj, date)?.count || 0],
        symbol: 'rect',
        itemStyle: {
          borderColor: '#00BFFF',
          borderWidth: 1,
          borderType: 'solid'
        }
      });
    }
    return data;
  }

  // 寻找不包含用户的日期，并对其绘制其他颜色。
  function highlightUnSelectedUsersDates(obj) {
    const initData = formatData(obj);
    const unselectedUsersDates = [];
    const maskData = [];
    _.forEach(initData, function (item) {
      if (item[1] === 0) {
        unselectedUsersDates.push(item[0])
      }
    })
    // 绘制面罩，以灰色的高亮图层的形式显示。
    for (const time of unselectedUsersDates) {
      dateFlag[time] = false; // 将不可选取的日子的flag设置为false,及不可选取
      maskData.push({
        value: [time, 0],
        symbol: 'rect',
        itemStyle: {
          color: 'rgba(119, 136, 153, 5)',
        },
        cursor: 'not-allowed', // 显示不可选取
        emphasis: {
          scale: false
        }
      });
    }
    return maskData;
  }

  // 自适应计算格网长宽
  const cellHeight = (bottomHeight - 10) / 8; //共8行，自适应计算
  const cellWidth = (bottomWidth - 140) / 53; //共53列，自适应计算
  const cellSize = [cellWidth, cellHeight]; // 日历单元格大小

  // 参数设置
  const option = {
    tooltip: {
      formatter: function (params) {// 说明某日出行用户数量
        return '日期: ' + params.value[0] + '<br />' + '出行用户: ' + params.value[1];
      },
    },
    visualMap: {
      calculable: true,
      orient: 'vertical',
      // left: 'bottom',
      top: 35,
      left: bottomWidth - 105,
      itemWidth: 5,
      itemHeight: 0,
      textStyle: {
        color: '#fff',
        fontSize: 12,
      },
      precision: 0,
      align: 'auto',
      formatter: function (value) {
        return parseInt(value)
      },
      handleIcon: 'circle'
    },
    calendar: {
      orient: 'horizontal',
      top: 30,
      cellSize: cellSize,
      range: year || +new Date().getFullYear(), // 日历图坐标范围(某一年)
      itemStyle: {
        borderWidth: 0.5
      },
      splitLine: {
        show: true,
        lineStyle: {
          width: 0.75,
        }
      },
      dayLabel: {
        color: '#fff',
        nameMap: 'cn',
      },
      monthLabel: {
        color: '#fff',
        nameMap: 'cn',
      },
      yearLabel: { show: false },
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
    }, {
      type: 'scatter',
      name: '面罩',
      coordinateSystem: 'calendar',
      symbolSize: cellSize,
      data: [],
      zlevel: 2,
    }]
  };

  // 初始化 ECharts 实例对象
  useEffect(() => {
    if (!ref.current) return () => {};
    myChart = echarts.init(ref.current);
    myChart.setOption(option);
    window.onresize = myChart.resize;
  }, [ref]);

  // strDate: yyyy-MM-dd
  function str2date(strDate) {
    strDate.replace('-', '/');
    return new Date(strDate);
  }

  // 组织日历数据
  function formatData(obj) {
    // const year = str2date(Object.keys(obj)[0]).getFullYear();
    let start = +echarts.number.parseDate(year + '-01-01');
    let end = +echarts.number.parseDate((+year + 1) + '-01-01');
    let dayTime = 3600 * 24 * 1000;
    let data = [];
    for (let time = start; time < end; time += dayTime) {
      const date = echarts.format.formatTime('yyyy-MM-dd', time);
      data.push([
        date,
        Reflect.get(obj, date)?.count || 0, // 没有数据用 0 填充
      ]);
    }
    return data;
  }

  // 将得到的数据重新数组，并重新渲染日历内容和位置、大小
  useEffect(() => {
    //data或rightWidth值改变后重新渲染
    const format = formatData(data);
    const counts = format.map(item => (item[1]));
    myChart.setOption({
      // prevProps获取到的bottomWidth/Height是0，
      // 在PageSelect页面componentDidMount获取到bottomWidth/Height值后，rightWidth值改变后重新渲染
      calendar: {
        left: cellWidth * 1.5,
        cellSize: cellSize,
      },
      visualMap: {
        // min: Math.min(...counts),
        // max: Math.max(...counts)
        min: 0,
        max: 500,
        itemHeight: bottomHeight - 50,
      },
      series: [{
        data: format,
      }, {
        name: '高亮',
        symbolSize: cellSize,
      }, {
        name: '面罩',
        symbolSize: cellSize
      }]
    })
  }, [data, bottomWidth, bottomHeight]);


  //返回所有筛选的用户
  function getUsers(obj, times) {
    let users = [];
    for (let i = 0; i < times.length; i++) { // 对每一段日期分别查找用户
      let start = +echarts.number.parseDate(times[i].start);
      let end = +echarts.number.parseDate(times[i].end);
      let dayTime = 3600 * 24 * 1000;
      for (let time = start; time <= end; time += dayTime) { // 对每段日期下的每一天分别查找用户
        const date = echarts.format.formatTime('yyyy-MM-dd', time);
        users.push(Reflect.get(obj, date).users.map(item => +item)); // 将每个日期下的用户加入到users数组中
      }
    }
    let finalUsers = users.reduce((prev, cur) => {
      if (prev.length === 0) return [...cur]; // 解决初始数组为[]的情况
      return Array.from(new Set(prev.filter(item => cur.includes(item)))) // 对每一个日期下的用户集合求交集
    }, [])
    return finalUsers;
  }

  // 记录框选的日期范围
  const [date, setDate] = useState({ start: '', end: '' });
  // 记录鼠标状态
  const [action, setAction] = useState(() => ({ mousedown: false, mousemove: false }));
  // 确保函数只执行一次
  const isdown = useRef(false);

  useEffect(() => {
    const wait = 50;
    if (!myChart) return () => {
    };
    // 鼠标按下事件
    const mouseDown = (params) => {
      //需要判断当前
      if (dateFlag[params.data[0]]) {
        // if (isdown.current) return;
        // 已触发，添加标记
        isdown.current = true;
        // params.data : (string | number)[] such as ['yyyy-MM-dd', 20]
        setAction(prev => {
          return {
            ...prev,
            mousedown: true,
          }
        });
        setDate({
          start: params.data[0] || params.data.value[0],
          end: params.data[0] || params.data.value[0],
        });
        // console.log('timePeriod_Down:', timePeriod)
      }
    };
    myChart.on('mousedown', mouseDown);
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
        ));
        setDate(prev => (
          {
            ...prev,
            end: params.data[0] || params.data.value[0],
          }
        ))
      },
      wait,
      false
    );
    const mouseMove = (params) => {
      action.mousedown && selectDate(params);
    };
    myChart.on('mousemove', mouseMove);

    // 鼠标抬起事件：结束选取
    const endSelect = (params) => {
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
      ) && ([start, end] = [end, start]);

      // 选择的日期范围
      // start: yyyy-MM-dd
      // end: yyyy-MM-dd
      // console.log(start, end);
      // 每次选择完则向timePeriod中添加本次筛选的日期，提供给下一次渲染。
      timePeriod.push({ start: start, end: end });
      // 返回筛选后符合要求的所有用户id信息，传递给其他页面。
      let userIDs = getUsers(data, timePeriod);
      //将数据传递到setSelectedByCalendar数组中
      dispatch(setSelectedByCalendar(userIDs));
    };
    const mouseUp = (params) => {
      if (isdown.current) { //如果点击的是不可选取的内容，则isdown不会变为true，也就不存在mouseUp功能
        endSelect(params)
      }
    };
    myChart.on('mouseup', mouseUp);

    return () => {
      myChart.off('mousedown', mouseDown);
      myChart.off('mousemove', mouseMove);
      myChart.off('mouseup', mouseUp);
    }
  }, [myChart, date, action]);

  // 高亮筛选部分
  useEffect(() => {
    if (!date.start || !date.end) return () => {
    };
    myChart?.setOption({
      series: [{
        name: '高亮',
        data: highLightData(data, date.start, date.end)
      }]
    });
  }, [data, date]);

  // 如果 data 改变，需要对不包含筛选用户的日期添加面罩作为提示
  useEffect(() => {
    initDateFlag();
    myChart?.setOption({
      series: [{
        name: '面罩',
        data: highlightUnSelectedUsersDates(data),
      }]
    })
  }, [data]);

  // 日历重置
  useEffect(() => {
    setTimeout(() => {
      myChart?.setOption({
        series: [{
          name: '高亮',
          data: [],
        }]
      });
    }, 600)
    //清空setSelectedByCalendar数组
    dispatch(setSelectedByCalendar([]));
    timePeriod = [];
  }, [calendarReload])

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