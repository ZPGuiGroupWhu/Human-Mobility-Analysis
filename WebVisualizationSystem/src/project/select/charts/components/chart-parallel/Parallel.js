import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
// react-redux
import { connect } from 'react-redux';
import { setSelectedByParallel } from '@/app/slice/selectSlice';


class Parallel extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
    }
  }

  ref = React.createRef();
  chart = null;

  // 坐标轴信息
  // schema = [
  //   { index: 0, text: '外向性' },
  //   { index: 1, text: '开放性' },
  //   { index: 2, text: '神经质性' },
  //   { index: 3, text: '尽责性' },
  // ];
  schema = this.props.keys.map((item, idx) => ({ index: idx, text: item }));
  newKeys = [...this.props.keys, '人员编号'];

  // 选框样式
  areaSelectStyle = {
    width: 15,
    borderWidth: .8,
    borderColor: 'rgba(160,197,232)',
    color: 'rgba(160,197,232)',
    opacity: .4,
  }

  // 线样式
  lineStyle = {
    width: 1,
    opacity: 0.5,
    cap: 'round',
    join: 'round',
  };

  option = {
    tooltip: {
      trigger: 'item', // 触发类型
      confine: true, // tooltip 限制在图表区域内
      formatter: (params) => {
        return `外向性：${(params.data[0]).toFixed(5)}<br/>
        开放性：${(params.data[1]).toFixed(5)}<br/>
        神经质性：${(params.data[2]).toFixed(5)}<br/>
        尽责性：${(params.data[3]).toFixed(5)}<br/>`
      }
    },
    // 工具栏配置
    toolbox: {
      feature: {
        brush: {
          title: {
            clear: '还原',
          },
          // 更换icon图标
          icon: {
            clear: "M819.199242 238.932954l136.532575 0c9.421335 0 17.066098-7.644762 17.067994-17.066098l0-136.532575-34.134092 0L938.665719 174.927029C838.326316 64.646781 701.016372 0 563.20019 0 280.88245 0 51.20019 229.682261 51.20019 512s229.682261 512 512 512c160.289736 0 308.325479-72.977903 406.118524-200.225656l-27.067616-20.78799c-91.272624 118.749781-229.445258 186.879554-379.050907 186.879554-263.509197 0-477.865908-214.356711-477.865908-477.865908S299.689097 34.134092 563.20019 34.134092c131.090991 0 262.003755 63.224764 356.406712 170.664771l-100.405764 0L819.201138 238.932954z",
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
      toolbox: ['clear'],
      xAxisIndex: 0,
      throttleType: 'debounce',
      throttleDelay: 300,
    },
    visualMap: [
      {
        type: 'piecewise',
        min: 0,
        max: 10,
        splitNumber: 6,
        show: false,
        seriesIndex: 0,
        dimension: 0,
        inRange: {
          color: ['#D00000', '#DC2F02', '#E85D04', '#F48C06', '#FAA307', '#FFBA08']
        }
      }
    ],
    parallelAxis: this.schema.map((item, idx) => ({
      dim: idx,
      name: item.text,
      areaSelectStyle: this.areaSelectStyle
    })),
    parallel: {
      left: '10%',
      right: '13%',
      bottom: '17%',
      top: '13%',
      parallelAxisDefault: {
        type: 'value',
        nameLocation: 'start',
        nameGap: 20,
        nameTextStyle: {
          fontSize: 12,
          color: '#fff',
        },
        min: 0,
        max: 8,
        axisLine: {
          lineStyle: {
            color: '#fff',
          }
        },
        axisTick: {
          show: false,
        },
        axisLabel: {
          formatter: (value) => (parseInt(value)),
        }
      }
    },
    series: [
      {
        name: '值',
        type: 'parallel',
        lineStyle: this.lineStyle,
        inactiveOpacity: 0.02,
        activeOpacity: 1,
        realtime: true,
        data: [],
      },
    ]
  };

  handleData = (data) => {
    return Object.values(data).map(item => {
      return this.newKeys.map((key, idx, arr) => item[key])
    })
  }

  // 存储刷选的数据索引映射
  onAxisAreaSelected = (params) => {
    let series0 = this.chart.getModel().getSeries()[0];
    let indices0 = series0.getRawIndicesByActiveState('active');
    const payload = indices0.map(item => this.state.data[item][this.newKeys.findIndex(key => key === '人员编号')]);
    this.props.setSelectedByParallel(payload);
    // 联动其他图层
    this.props.handleBrushEnd();
  }


  componentDidMount() {
    this.chart = echarts.init(this.ref.current); // 初始化容器
    this.chart.setOption(this.option); // 初始化视图
    this.chart.on('axisareaselected', this.onAxisAreaSelected);
  }

  componentDidUpdate(prevProps, prevState) {
    if (!_.isEqual(prevProps.data, this.props.data)) {
      this.setState({
        data: this.handleData(this.props.data),
      })
    }

    if (!_.isEqual(prevState.data, this.state.data)) {
      Reflect.set(this.option.series[0], 'data', this.state.data);
      setTimeout(() => { // 延迟缓冲，避免坐标轴框选显示问题
        this.chart.setOption(this.option);
      }, 800)
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
    setSelectedByParallel: (payload) => dispatch(setSelectedByParallel(payload)),
  }
}

export default connect(null, mapDispatchToProps)(Parallel);