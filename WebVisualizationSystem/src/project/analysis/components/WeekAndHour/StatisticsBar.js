import React, {useRef, useEffect} from 'react';
import * as echarts from 'echarts';
import './CalendarWindow.scss';

// hour、weekday的统计柱状图，参数 
export default function StatisticsBar(props) {
  const {
    type='day',
    grid,
    xData,
    data,
    // acIdx,
  } = props;
  const ref = useRef(null);
  const myChart = useRef(null);

  const option = {
    // 位置
    grid,
    // 轴配置
    xAxis: {
      type: 'category',
      data: xData,
      boundaryGap: true,
      // 标签
      axisLabel: {
        color: '#fff', // 颜色
        align: 'center', // 位置
        alignWithLabel: true,
        interval: 0, // 间隔
      },
      axisTick: {
        interval: 0,
      },
      max: xData.length - 1,
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#fff',
        align: 'center',
        margin: 15,
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
          borderRadius: [5, 5, 0 ,0],  // 柱状圆角半径
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
