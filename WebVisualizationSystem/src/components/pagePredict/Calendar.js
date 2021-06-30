import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { getCalendarByCount, selectByTime } from '@/network';
import { debounce } from '@/common/func/debounce';

let myChart = null;
export default function Calendar(props) {
  const { callback: { setByTime } } = props;
  const ref = useRef(null);

  // 伪数据
  const maxValue = 100;
  const minValue = 0;
  const year = '2018';
  // 数据格式中，必须包含 'yyy-MM-dd' 与额外数据
  function getVirtualData(year) {
    year = year || '2018';
    let date = +echarts.number.parseDate(year + '-01-01');
    let end = +echarts.number.parseDate((+year + 1) + '-01-01');
    let dayTime = 3600 * 24 * 1000;
    let data = [];
    for (let time = date; time < end; time += dayTime) {
      data.push([
        echarts.format.formatTime('yyyy-MM-dd', time),
        Math.floor(Math.random() * 100)
      ]);
    }
    return data;
  }

  // strDate: yyyy-MM-dd
  function str2date(strDate) {
    strDate.replace('-', '/');
    return new Date(strDate);
  }

  // 数据格式中，必须包含 'yyy-MM-dd' 与额外数据
  function formatData(obj) {
    const year = str2date(Object.keys(obj)[0]).getFullYear();
    let start = +echarts.number.parseDate(year + '-01-01');
    let end = +echarts.number.parseDate((+year + 1) + '-01-01');
    let dayTime = 3600 * 24 * 1000;
    let data = [];
    for (let time = start; time < end; time += dayTime) {
      const date = echarts.format.formatTime('yyyy-MM-dd', time)
      data.push([
        date,
        Reflect.get(obj, date) || undefined // 没有数据用 undefined 填充
      ]);
    }
    return data;
  }

  const option = {
    // title: {
    //   top: 10,
    //   left: 'center',
    //   text: year + '年用户出行统计',
    // },
    tooltip: {},
    visualMap: {
      min: minValue,
      max: maxValue,
      calculable: true,
      orient: 'vertical',
      left: 'right',
      top: 'top',
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
      cellSize: ['auto', 16],
      range: year, // 日历图坐标范围(某一年)
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
    series: {
      type: 'heatmap',
      coordinateSystem: 'calendar',
      data: []
    }
  }

  const [data, setData] = useState(null);

  useEffect(() => {
    getCalendarByCount().then(
      res => setData(res)
    ).catch(
      err => console.log(err)
    )
  }, [])

  // 初始化 echarts 实例对象
  useEffect(() => {
    if (!ref.current) return () => { };
    myChart = echarts.init(ref.current);
    myChart.setOption(option);
  }, [ref])

  useEffect(() => {
    if (!data) {
      myChart.setOption({
        series: {
          data: getVirtualData(2018),
        }
      })
    } else {
      const format = formatData(data);
      myChart.setOption({
        visualMap: {
          min: Math.min(...Object.values(data)),
          max: Math.max(...Object.values(data))
        },
        series: {
          data: format,
        }
      })
    }
  }, [data])


  // 初始化鼠标状态
  function initAction() {
    return {
      mousedown: false,
      mousemove: false,
    }
  }
  const [date, setDate] = useState({ start: '', end: '' });
  const [action, setAction] = useState(initAction);
  useEffect(() => {
    const wait = 100;

    if (!myChart) return () => { };
    // 鼠标按下事件
    myChart.on('mousedown', (params) => {
      // params.data : (string | number)[] such as ['yyyy-MM-dd', 20]
      setAction(prev => {
        return {
          ...prev,
          mousedown: true,
        }
      })
      setDate({
        start: params.data[0],
        end: params.data[0],
      })
    })

    // 鼠标移动事件
    const selectDate = debounce(
      (params) => {
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
            end: params.data[0]
          }
        ))
      },
      wait,
      false
    )
    const mouseMove = (params) => {
      action.mousedown && selectDate(params)
    }
    myChart.on('mousemove', mouseMove);

    // 鼠标抬起事件
    const mouseUp = (params) => {
      // 重置鼠标状态
      setAction(initAction);

      let start = date.start, end = date.end;
      let startDate = str2date(start), endDate = str2date(end);
      // 校正时间顺序
      (
        (startDate.getMonth() > endDate.getMonth()) ||
        (startDate.getDay() > endDate.getDay())
      ) && ([start, end] = [end, start])

      // 依据筛选条件发送请求
      selectByTime(start, end).then(
        res => {
          // 将接收到的数据更新到 PagePredict 页面 state 中管理
          setByTime(res || [])
        }
      ).catch(
        err => console.log(err)
      );
    }
    myChart.on('mouseup', mouseUp)

    return () => {
      myChart.off('mousemove', mouseMove);
      myChart.off('mouseup', mouseUp);
    }
  }, [myChart, date, action])

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
