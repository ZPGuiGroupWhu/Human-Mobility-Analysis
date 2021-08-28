import React, { Component } from 'react';
import * as echarts from 'echarts'

class Bar extends Component {
  constructor(props) {
    super(props);
    // props
    this.yAxisName = props.yAxisName;
    this.height = props.height;
    // state
    this.state = {}
  }

  ref = React.createRef(null);
  chart = null;

  option = {
    tooltip: {
      show:true,
      trigger: 'axis',
    },
    // grid - 定位图表在容器中的位置
    grid: {
      show: true, // 是否显示直角坐标系网格
      left: '40', // 距离容器左侧距离
      top: '10', // 距离容器上侧距离
      right: '10',
      bottom: '10',
    },
    xAxis: {
      show: false,
      type: 'category'
    },
    yAxis: {
      show: true,
      gridIndex: 0,
      position: 'left',
      type: 'value',
      name: this.yAxisName, // 坐标轴名称
      nameLocation: 'end', // 坐标周名称显示位置
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
      },
      nameGap: 15, // 坐标轴名称与轴线距离
      min: 'dataMin', // 刻度最小值
      max: 'dataMax', // 刻度最大值
      axisLine: {
        show: true, // 是否显示坐标轴线
        symbol: ['none', 'arrow'],
        symbolSize: [5,8],
        lineStyle: {
          color: '#fff',
        }
      },
      axisTick: {
        show: false, // 是否显示坐标刻度
      },
      axisLabel: {
        show: true, // 是否显示坐标轴刻度标签
        rotate: 0, // 刻度标签旋转角度
        margin: 8, // 刻度标签与轴线距离
        color: '#fff', // 刻度标签文字颜色
        fontSize: 12, // 刻度标签文字大小
        formatter: (value, index) => (value.toFixed(3)),
      }
    },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: 0,
        filterMode: 'filter', // 过滤模式
        // --- 百分比 ---
        // start: 0, // 初始起始范围
        // end: 4, // 初始结束范围
        // minSpan: 10, // 最小取值窗口(%)
        // maxSpan: 20, // 最大取值窗口(%)
        // --- 实值 ---
        startValue: 0,
        endValue: 6,
        minValueSpan: 6,
        maxValueSpan: 50,
        // -----------
        orient: 'horizontal', // 布局方式
        zoomLock: false, // 是否锁定取值窗口(若锁定，只能平移不能缩放)
        zoomOnMouseWheel: true, // 触发缩放的条件
        moveOnMouseMove: true, // 触发平移的条件
      }, 
      {
        type: 'slider',
        yAxisIndex: 0,
        filterMode: 'filter', // 过滤模式
        textStyle: {
          color: '#fff',
          fontSize: 12,
        },
        // --- 百分比 ---
        start: 0, // 初始起始范围
        end: 100, // 初始结束范围
        minSpan: 0, // 最小取值窗口(%)
        maxSpan: 100, // 最大取值窗口(%)
        // --- 实值 ---
        // startValue: 0,
        // endValue: 6,
        // minValueSpan: 6,
        // maxValueSpan: 50,
        // -----------
        orient: 'vertical', // 布局方式
        zoomLock: false, // 是否锁定取值窗口(若锁定，只能平移不能缩放)
      }
    ],
    series: [
      {
        type: 'bar',
        showBackground: true, // 是否显示柱条背景颜色
        backgroundStyle: {
          color: 'rgba(180, 180, 180, 0.2)', // 柱条背景颜色
          borderColor: '#000', // 描边颜色
          borderWidth: 0, // 描边宽度
        },
        itemStyle: {
          color: new echarts.graphic.LinearGradient(
            0, 0, 0, 1,
            [
              { offset: 0, color: '#83bff6' },
              { offset: 0.5, color: '#188df0' },
              { offset: 1, color: '#188df0' }
            ]
          ), // 柱条颜色
          borderColor: '#000', // 描边颜色
          borderWidth: 0, // 描边宽度
          borderRadius: [5, 5, 0 , 0], // 描边弧度
        },
        emphasis: {
          focus: 'series', // 聚焦效果
          blurScope: 'coordinateSystem', // 淡出范围
          itemStyle: {
            color: new echarts.graphic.LinearGradient(
              0, 0, 0, 1,
              [
                { offset: 0, color: '#63b2ee' },
                { offset: 0.7, color: '#efa666' },
                { offset: 1, color: '#f89588' }
              ]
            )
          }
        },
        large: true, // 是否开启大数据量优化
        largeThreshold: 400, // 开启优化的阈值
        data: [],
      }
    ]
  };

  componentDidMount() {
    this.chart = echarts.init(this.ref.current);
    this.chart.setOption(this.option);
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevProps.data !== this.props.data) {
      this.chart.setOption(
        {
          series: [{
            data: this.props.data
          }]
        }
      )
    }
  }

  render() {
    return (
      <div
        style={{
          height: this.height,
        }}
        ref={this.ref}
      ></div>
    );
  }
}

export default Bar;