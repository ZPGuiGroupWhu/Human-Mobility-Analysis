import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
import Store from '@/store';

class Scatter extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }

  ref = React.createRef();
  chart = null;

  option = {
    // 工具栏配置
    toolbox: {
      iconStyle: {
        color: '#fff', // icon 图形填充颜色
        borderColor: '#fff', // icon 图形描边颜色
      },
      emphasis: {
        iconStyle: {
          color: '#7cd6cf',
          borderColor: '#7cd6cf',
        }
      }
    },
    // 框选工具配置
    brush: {
      toolbox: ['rect', 'keep', 'clear'],
      xAxisIndex: 0,
      throttleType: 'debounce',
    },
    tooltip: {
      show: true,
      trigger: 'item', // 触发类型
      confine: true, // tooltip 限制在图表区域内
      axisPointer: {
        type: 'cross',
        snap: true, // 指示器是否自动吸附
        label: {
          show: false,
        },
        crossStyle: {
          color: '#fff',
        }
      },
      formatter: (params) => {
        return `人员编号: ${params.value[2]}<br/>${this.props.xAxisName}: ${params.value[0].toFixed(3)}<br/>${this.props.yAxisName}: ${params.value[1].toFixed(3)}`
      }
    },
    // grid - 定位图表在容器中的位置
    grid: {
      show: false, // 是否显示直角坐标系网格
      left: '45', // 距离容器左侧距离
      top: '40', // 距离容器上侧距离
      right: '20',
      bottom: '40',
    },
    xAxis: {
      name: this.props.xAxisName, // 坐标轴名称
      nameLocation: 'center', // 坐标轴名称显示位置
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
        lineHeight: 25,
      },
      splitLine: { show: false }, // 坐标轴区域的分隔线
      show: true,
      gridIndex: 0,
      position: 'bottom',
      type: 'value',
      nameGap: 15, // 坐标轴名称与轴线距离
      min: 'dataMin', // 刻度最小值
      max: 'dataMax', // 刻度最大值
      axisLine: {
        show: true, // 是否显示坐标轴线
        symbol: ['none', 'arrow'],
        symbolSize: [5, 8],
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
        formatter: (value, index) => (value.toFixed(1)),
      }
    },
    yAxis: {
      name: this.props.yAxisName, // 坐标轴名称
      nameLocation: 'center', // 坐标轴名称显示位置
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
        lineHeight: 40,
      },
      splitLine: { show: false }, // 坐标轴区域的分隔线
      show: true,
      gridIndex: 0,
      position: 'left',
      type: 'value',
      nameGap: 15, // 坐标轴名称与轴线距离
      min: 'dataMin', // 刻度最小值
      max: 'dataMax', // 刻度最大值
      axisLine: {
        show: true, // 是否显示坐标轴线
        symbol: ['none', 'arrow'],
        symbolSize: [5, 8],
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
        formatter: (value, index) => (value.toFixed(1)),
      }
    },
    visualMap: [{
      show: false,
      type: 'continuous', // 视觉映射类型
      min: 0, // 视觉最小映射
      max: 2, // 视觉最大映射
      dimension: 3, // 映射维度
      inRange: {
        color: [
          '#00E0FF',
          '#74F9FF',
          '#A6FFF2',
          '#E8FFE8',
        ]
      },
      outOfRange: {
        color: ['#ccc']
      },
      textStyle: {
        color: '#fff'
      },
    }],
    series: [
      {
        zlevel: 1,
        type: 'scatter',
        symbol: 'circle', // 标记图形
        symbolSize: 5, // 标记大小
        emphasis: {
          focus: 'series',
        },
        data: [],
        animationThreshold: 5000,
        progressiveThreshold: 5000
      }
    ],
    animationEasingUpdate: 'cubicInOut',
    animationDurationUpdate: 2000
  };

  // 存储刷选的数据索引映射
  onBrushSelected = (params) => {
    let brushComponent = params.batch[0];
    this.context.dispatch({
      type: 'setSelectedUsers',
      payload: brushComponent.selected[0].dataIndex.map(item => this.props.data[item][2]), // 刷选索引映射到数据维度
    });
  }

  onBrushEnd = (params) => {
    this.props.handleBrushEnd();
  }

  componentDidMount() {
    this.chart = echarts.init(this.ref.current); // 初始化容器
    this.chart.setOption(this.option); // 初始化视图
    this.chart.getZr().configLayer(1, { motionBlur: 0.5 }); // zlevel 为 1 的层开启尾迹特效
    this.chart.on('brushSelected', this.onBrushSelected); // 添加 brushSelected 事件
    this.chart.on('brushEnd', this.onBrushEnd);
  }

  componentDidUpdate(prevProps, prevState) {
    if (!_.isEqual(prevProps.data, this.props.data)) {
      this.option.series[0].data = this.props.data;
      Reflect.set(this.option.xAxis, 'name', this.props.xAxisName);
      Reflect.set(this.option.yAxis, 'name', this.props.yAxisName);
      this.chart.setOption(this.option);
    }

    if (this.props.withFilter) {
      if ((prevProps.xAxisName !== this.props.xAxisName) || (prevProps.yAxisName !== this.props.yAxisName)) {
        this.chart.dispatchAction({ type: 'brush', areas: [] }); // 清除框选
      }
    }
  }

  render() {
    return (
      <div
        style={{
          height: this.props.height,
        }}
        draggable={false}
        ref={this.ref}
      ></div>
    );
  }
}

Scatter.contextType = Store;

export default Scatter;