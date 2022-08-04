import React, { useRef, useEffect } from 'react';
import * as echarts from 'echarts';
import './CalendarWindow.scss';

// hour、weekday的统计柱状图，参数 
export default function StatisticsBar(props) {
  const {
    type = 'day',
    grid,
    xData,
    data,
    isDay, // 判断是 天 还是 周，采用不同样式坐标轴
    // acIdx,
  } = props;
  const ref = useRef(null);
  const myChart = useRef(null);

  const option = {
    // 位置
    grid,
    // tooltips
    tooltip: {
      formatter: function (params) {// 说明某日出行用户数量
        return  params.name + '  出行次数: ' + params.data;
      },
    },
    // 轴配置
    xAxis: {
      type: 'category',
      data: xData,
      boundaryGap: true,
      // 标签
      axisLabel: {
        show: true,
        interval: (isDay ? 1 : 0), // 间隔几个显示一次label
        inside: false, // label显示在外侧
        margin: 10,
        color: "#ccc",
        fontStyle: "normal",
        fontWeight: "bold",
        fontSize: 12
      },
      axisTick: { // 坐标轴刻度线style
        alignWithLabel: true,
        interval: 0,
        inside: false,
        length: 3
      },
      max: xData.length - 1,
    },
    yAxis: {
      type: 'value',
      name: '出行次数',
      nameLocation: 'center', // 坐标轴名称显示位置
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
      },
      nameGap: 28, // 坐标轴名称与轴线距离
      axisLabel: {
        color: '#fff',
        align: 'center',
        margin: 13,
      },
    },
    series: [
      {
        data,
        type: 'bar',
        // 柱状背景及配色
        showBackground: true,
        backgroundStyle: {
          color: 'rgba(180, 180, 180, 0.2)'
        },
        // 柱状样式
        itemStyle: {
          color: '#40FFD9',  // 柱状颜色
          borderRadius: [5, 5, 0, 0],  // 柱状圆角半径
        },

      }
    ]
  };

  // 初始化柱状图
  useEffect(() => {
    myChart.current = echarts.init(ref.current);
    myChart.current.setOption(option)
    return () => {
      myChart.current.dispose();
    }
  }, [])

  // 重新加载数据
  useEffect(() => {
    option.series[0].data = data
    myChart.current.setOption(option)
  }, [data])


  // 高亮柱状图中的某一个柱子
  // useEffect(() => {
  //   let empData = option.series[0].data.map((item, idx) => {
  //     return (idx === acIdx - 1) ? {value: item, itemStyle: {color: '#F46400'}} : item
  //   })
  //   option.series[0].data = empData
  //   myChart.current.setOption(option);
  // }, [acIdx])

  return (
    <div ref={ref} className={`statistics-${type}bar`}></div>
  )
}
