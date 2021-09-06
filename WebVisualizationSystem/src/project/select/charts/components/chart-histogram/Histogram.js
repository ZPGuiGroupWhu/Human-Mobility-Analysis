import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
import Store from '@/store';

class Histogram extends Component {
  ref = React.createRef(null);
  chart = null;

  yAxisName = '人数';
  option = {
    tooltip: {
      show: true,
      trigger: 'axis',
      confine: true, // tooltip 限制在图表区域内
      formatter: (params) => {
        // console.log(params);
        return `区间范围：${params[0].data[0]}<br/>人数：${params[0].data[1]}`
      }
    },
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
      toolbox: ['lineX', 'lineY', 'keep', 'clear'],
      xAxisIndex: 0,
      throttleType: 'debounce',
    },
    // grid - 定位图表在容器中的位置
    grid: {
      show: true, // 是否显示直角坐标系网格
      left: '35', // 距离容器左侧距离
      top: '40', // 距离容器上侧距离
      right: '20',
      bottom: '35',
    },
    xAxis: {
      show: true,
      name: this.props.xAxisName,
      type: 'category',
      nameLocation: 'center',
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
      },
      nameGap: 15, // 坐标轴名称与轴线距离
      axisLabel: {
        show: false, // 是否显示坐标轴刻度标签
        rotate: 0, // 刻度标签旋转角度
        margin: 8, // 刻度标签与轴线距离
        color: '#fff', // 刻度标签文字颜色
        fontSize: 12, // 刻度标签文字大小
      }
    },
    yAxis: {
      show: true,
      name: this.yAxisName,
      gridIndex: 0,
      position: 'left',
      type: 'value',
      nameLocation: 'end', // 坐标轴名称显示位置
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
      },
      nameGap: 15, // 坐标轴名称与轴线距离
      axisLine: {
        show: true, // 是否显示坐标轴线
        symbol: ['none', 'none'],
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
      }
    },
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
          borderRadius: [5, 5, 0, 0], // 描边弧度
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
    ],
  };

  constructor(props) {
    super(props);
    this.state = {
      data: null,
    }
  }

  // 数据预处理
  handleData = (data, dim) => {
    data.sort((a, b) => (a[dim] - b[dim]));
    return function (gap = 0.1) {
      let res = [];
      const times = Math.ceil(1 / gap);
      for (let i = 1; i < times + 1; i++) {
        const min = (i - 1) * gap;
        const max = i * gap;
        const newData = data.filter(item => {
          return (item[0] >= min && item[0] < max)
        })
        res.push([`${min.toFixed(1)}-${max.toFixed(1)}`, newData.length, newData]);
      }
      return res;
    }
  }

  // 存储刷选的数据索引映射
  onBrushSelected = (params) => {
    let brushComponent = params.batch[0];
    if (this.props.withFilter && !brushComponent.selected[0].dataIndex.length) return; // 若开启过滤，则始终保留历史刷选数据
    this.context.dispatch({
      type: 'setSelectedUsers',
      payload: brushComponent.selected[0].dataIndex.map(item => {
        return this.state.data[item][2].map(item => item[2]); // dim=3: 人员编号
      }).flat(Infinity), // 刷选索引映射到数据维度
    });
  }

  onBrushEnd = (params) => {
    this.props.handleBrushEnd();
  }

  componentDidMount() {
    this.chart = echarts.init(this.ref.current); // 初始化容器
    this.chart.setOption(this.option); // 初始化视图
    this.chart.on('brushSelected', this.onBrushSelected); // 添加 brushSelected 事件
    this.chart.on('brushEnd', this.onBrushEnd);
  }

  componentDidUpdate(prevProps, prevState) {
    if (!this.props.data) return;
    // 数据生成
    if (!_.isEqual(prevProps.data, this.props.data)) {
      this.setState({
        data: this.handleData(this.props.data, 0)(),
      });
    }
    // 数据驱动
    if (!_.isEqual(prevState.data, this.state.data)) {
      Reflect.set(this.option.series[0], 'data', this.state.data);
      this.chart.setOption(this.option);
    }

    if (prevProps.xAxisName !== this.props.xAxisName) {
      Reflect.set(this.option.xAxis, 'name', this.props.xAxisName);
      this.chart.setOption(this.option);
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

Histogram.contextType = Store;

export default Histogram;