import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as echarts from 'echarts'; // ECharts
import PropTypes from 'prop-types';

function RelationChart(props) {
  const {
    titleText, // 图表标题，例如 '速度/转向角关系图'
    legendData, // 图例数据项名称，例如 ['速度', '转向角']
    xAxisData, // x轴刻度
    yAxis, // y轴标签名，例如 ['速度(km/h)', '转向角(rad)']
    data, // series data 数据项
  } = props;
  const ref = useRef(null); // 容器对象
  const myChart = useRef(null); // ECharts实例
  const relationChartRefCallback = useCallback(node => {
    if (node !== null) {
      ref.current = node;
      // let observer = new MutationObserver((mutationsList) => {
      //   const dom = mutationsList[0].target;
      //   let obj = dom.getBoundingClientRect();
      //   console.log(obj);
      //   node.style.left = (obj.x + obj.width) + 'px';
      //   node.style.top = (obj.y + obj.height) + 'px';
      // });
      // observer.observe(document.querySelector('#poi-frame'), {attributes: true});
    }
  }, [])

  // 静态配置项
  const option = {
    // 全局字体样式
    textStyle: {
      color: '#fff', // 字体颜色
    },
    title: {
      show: false,
      text: titleText,
      right: 15,
      top: 15,
      textStyle: {
        color: '#fff',
        fontSize: 15,
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        animation: true
      },
      formatter: (params) => {
        // console.log(params);
        return `${yAxis[1]}: ${params[0].value.toFixed(2)}<br/>${yAxis[0]}: ${params[1].value.toFixed(2)}`
      }
    },
    legend: {
      show: false,
      data: legendData,
      left: 10,
      top: 15,
      textStyle: {
        color: '#fff',
      }
    },
    axisPointer: {
      link: { xAxisIndex: 'all' }
    },
    dataZoom: [
      {
        type: 'inside',
        show: true,
        realtime: true,
        start: 0,
        end: 100,
        xAxisIndex: [0, 1]
      },
      {
        type: 'inside',
        realtime: true,
        start: 0,
        end: 100,
        xAxisIndex: [0, 1]
      }
    ],
    grid: [{
      left: 30,
      right: 10,
      top: '18%',
      height: '30%',
    }, {
      left: 30,
      right: 10,
      top: '53%',
      height: '30%',
    }],
    xAxis: [
      {
        type: 'category',
        show: false,
        boundaryGap: false,
        axisLine: { onZero: true },
        data: xAxisData,
      },
      {
        show: false,
        gridIndex: 1,
        type: 'category',
        boundaryGap: false,
        axisLine: { onZero: true },
        data: xAxisData,
        position: 'top',
      }
    ],
    yAxis: [
      {
        name: yAxis[0],
        type: 'value',
        splitLine: {
          lineStyle: {
            color: 'rgba(255, 255, 255, 0.5)',
            type: 'dashed',
          }
        }
        // max: 25
      },
      {
        gridIndex: 1,
        name: yAxis[1],
        type: 'value',
        inverse: true,
        splitLine: {
          lineStyle: {
            color: 'rgba(255, 255, 255, 0.5)',
            type: 'dashed',
          }
        }
      }
    ],
    series: [
      {
        name: legendData[0],
        type: 'line',
        lineStyle: {
          width: 1,
        },
        coordinateSystem: 'cartesian2d',
        symbolSize: 3,
        hoverAnimation: true,
        data: [],
      },
      {
        name: legendData[1],
        type: 'line',
        lineStyle: {
          width: 1,
        },
        coordinateSystem: 'cartesian2d',
        xAxisIndex: 1,
        yAxisIndex: 1,
        symbolSize: 3,
        hoverAnimation: true,
        data: [],
      }
    ],
  }

  useEffect(() => {
    myChart.current = echarts.init(ref.current);
    return () => {
      myChart.current.dispose();
      myChart.current = null;
    }
  }, [])

  const [idx, setIdx] = useState(-1);
  useEffect(() => {
    function getHighlightIndex(params) {
      // console.log(params);
      const dataIndex = params.batch?.[0].dataIndex || -1; // 高亮的数据索引
      setIdx(dataIndex);
    }
    function clearHighlightIndex() {
      setIdx(-1);
    }
    myChart.current.on('highlight', 'series', getHighlightIndex); // 添加高亮事件
    myChart.current.on('downplay', 'series', clearHighlightIndex);

    return () => {
      myChart.current?.off('highlight', getHighlightIndex); // 卸载高亮事件
      myChart.current?.off('downplay', clearHighlightIndex);
    }
  }, [])

  useEffect(() => {
    if (typeof props.onHighlight === 'function') {
      props.onHighlight(idx);
    }
  }, [idx])


  // 判空
  function handleTypeJudge(data) {
    let initBool = true;
    initBool = initBool && Array.isArray(data) && (!!(data[0]?.length)) && (!!(data[1]?.length))
    return initBool
  }

  // 数据驱动渲染视图
  useEffect(() => {
    if (handleTypeJudge(data)) {
      const chart = myChart.current;
      Reflect.set(option.series[0], 'data', data[0]);
      Reflect.set(option.series[1], 'data', data[1]);
      chart.setOption(option);
    }
  }, [data])


  return (
    <div
      ref={relationChartRefCallback}
      style={{
        width: '170px',
        height: '220px',
        borderRadius: '5px',
        backgroundColor: 'rgba(0, 0, 0, .7)',
        ...props.style,
      }}
    ></div>
  )
}

RelationChart.propTypes = {
  titleText: PropTypes.string,
  legendData: PropTypes.arrayOf(PropTypes.string),
  xAxisData: PropTypes.array,
  yAxis: PropTypes.arrayOf(PropTypes.string),
  data: PropTypes.array,
}

export default RelationChart;