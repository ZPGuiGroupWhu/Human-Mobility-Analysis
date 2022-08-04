import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
// react-redux
import { connect } from 'react-redux';
import { setSelectedByScatter } from '@/app/slice/selectSlice';


class Scatter extends Component {
  constructor(props) {
    super(props);
    this.state = {
      brushReload: true
    }
  }

  ref = React.createRef();
  chart = null;

  option = {
    // 工具栏配置
    toolbox: {
      feature: {
        // 基础框选
        brush: {
          title: {
            rect: '矩形选择',
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
            // 清除顶层的selectedByScatter数组
            this.props.setSelectedByScatter([]);
            // 用于清除顶层的selectedByScatter数组
            this.setState({
              brushReload: {}
            });
          }
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
      toolbox: ['rect', 'keep'],
      xAxisIndex: 0,
      throttleType: 'debounce',
      throttleDelay: 300,
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
    if (!brushComponent.selected[0].dataIndex.length) return; // 若开启过滤，则始终保留历史刷选数据
    const payload = brushComponent.selected[0].dataIndex.map(item => this.props.data[item][2]);
    this.props.setSelectedByScatter(payload);
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
      // 清除选框
      this.chart.dispatchAction({
        type: 'brush',
        areas: [],
      })
      this.option.series[0].data = this.props.data;
      Reflect.set(this.option.xAxis, 'name', this.props.xAxisName);
      Reflect.set(this.option.yAxis, 'name', this.props.yAxisName);
      this.chart.setOption(this.option);
    }

    // 单个 自身的还原按钮 清除
    if (prevState.brushReload !== this.state.brushReload) {
      // 清除2D地图上的选框
      this.chart.dispatchAction({
        type: 'brush',
        areas: [], // 点击reload同时清除选择框
      });
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
    setSelectedByScatter: (payload) => dispatch(setSelectedByScatter(payload)),
  }
}


export default connect(null, mapDispatchToProps)(Scatter);