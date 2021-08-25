import React, { useRef, useEffect } from 'react';
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

  useEffect(() => {
    myChart.current = echarts.init(ref.current);
    return () => {
      myChart.current.dispose();
      myChart.current = null;
    }
  }, [])

  useEffect(() => {
    if (!!data.length) {
      const chart = myChart.current;
      chart.setOption({
        // 全局字体样式
        textStyle: {
          color: '#fff', // 字体颜色
        },
        title: {
          text: titleText,
          right: 15,
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
        },
        legend: {
          data: legendData,
          left: 10,
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
          left: 40,
          right: 20,
          height: '30%'
        }, {
          left: 40,
          right: 20,
          top: '58%',
          height: '30%'
        }],
        xAxis: [
          {
            type: 'category',
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
            coordinateSystem: 'cartesian2d',
            symbolSize: 5,
            hoverAnimation: true,
            data: data[0],
          },
          {
            name: legendData[1],
            type: 'line',
            coordinateSystem: 'cartesian2d',
            xAxisIndex: 1,
            yAxisIndex: 1,
            symbolSize: 5,
            hoverAnimation: true,
            data: data[1],
          }
        ],
      })
    }
  }, [data])

  return (
    <div
      ref={ref}
      style={{
        width: '320px',
        height: '320px',
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