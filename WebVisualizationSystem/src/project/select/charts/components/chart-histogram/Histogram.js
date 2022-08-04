import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
// react-redux
import { connect } from 'react-redux';
import { setSelectedByHistogram } from '@/app/slice/selectSlice';

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
      feature: {
        // 基础框选
        brush: {
          title: {
            lineX: '横轴框选',
            lineY: '纵轴框选',
            keep: '开启多选',
          }
        },
        // 清除还原 功能
        myTool1: {
          show: true,
          title: '还原',
          icon:
            "M819.199242 238.932954l136.532575 0c9.421335 0 17.066098-7.644762 17.067994-17.066098l0-136.532575-34.134092 0L938.665719 174.927029C838.326316 64.646781 701.016372 0 563.20019 0 280.88245 0 51.20019 229.682261 51.20019 512s229.682261 512 512 512c160.289736 0 308.325479-72.977903 406.118524-200.225656l-27.067616-20.78799c-91.272624 118.749781-229.445258 186.879554-379.050907 186.879554-263.509197 0-477.865908-214.356711-477.865908-477.865908S299.689097 34.134092 563.20019 34.134092c131.090991 0 262.003755 63.224764 356.406712 170.664771l-100.405764 0L819.201138 238.932954z",
          onclick: () => {
            // 清除顶层的selectedByHistogram数组
            this.props.setSelectedByHistogram([]);
            // 标记reload
            this.setState({
              brushReload: {}
            });
          },
        }
      },
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
      toolbox: ['lineX', 'lineY', 'keep'],
      xAxisIndex: 0,
      throttleType: 'debounce',
      throttleDelay: 300,
    },
    // grid - 定位图表在容器中的位置
    grid: {
      show: true, // 是否显示直角坐标系网格
      left: '45', // 距离容器左侧距离
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
      nameLocation: 'center', // 坐标轴名称显示位置
      nameTextStyle: {
        color: '#fff', // 文本颜色
        fontSize: 12, // 文本大小
      },
      nameGap: 30, // 坐标轴名称与轴线距离
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
      brushReload: {}
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
    if (!brushComponent.selected[0].dataIndex.length) return; // 若开启过滤，则始终保留历史刷选数据
    const payload = brushComponent.selected[0].dataIndex.map(item => {
      return this.state.data[item][2].map(item => item[2]); // dim=3: 人员编号
    }).flat(Infinity) // 刷选索引映射到数据维度
    this.props.setSelectedByHistogram(payload);
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

    // 单个 自身的还原按钮 清除
    if (prevState.brushReload !== this.state.brushReload) {
      // 清除选框
      this.chart.dispatchAction({
        type: 'brush',
        areas: [],
      })
    }

    /**
     * 涉及到this.props.isReload的整体清除 => 此部分暂时保留，以供后续需要
     */
    
    // 整体清除
    if (prevProps.isReload !== this.props.isReload) {
      // 清除选框
      this.chart.dispatchAction({
        type: 'brush',
        areas: [],
      })
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

const mapDispatchToProps = (dispatch) => {
  return {
    setSelectedByHistogram: (payload) => dispatch(setSelectedByHistogram(payload)),
  }
}

export default connect(null, mapDispatchToProps)(Histogram);