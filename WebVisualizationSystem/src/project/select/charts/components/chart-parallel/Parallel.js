import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
import Store from '@/store';

class Parallel extends Component {
  static contextType = Store;

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
  schema = this.props.keys.map((item, idx) => ({index: idx, text: item}));
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
      toolbox: ['clear'],
      xAxisIndex: 0,
      throttleType: 'debounce',
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
    parallelAxis: [
      { dim: 0, name: this.schema[0].text, areaSelectStyle: this.areaSelectStyle, },
      { dim: 1, name: this.schema[1].text, areaSelectStyle: this.areaSelectStyle, },
      { dim: 2, name: this.schema[2].text, areaSelectStyle: this.areaSelectStyle, },
      { dim: 3, name: this.schema[3].text, areaSelectStyle: this.areaSelectStyle, },
    ],
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
        min: 'dataMin',
        max: 'dataMax',
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
        data: [],
      },
    ]
  };

  handleData = (data) => {
    return Object.values(data).map(item => {
      return this.newKeys.map(key => item[key])
    })
  }

  // 存储刷选的数据索引映射
  onAxisAreaSelected = (params) => {
    var series0 = this.chart.getModel().getSeries()[0];
    var indices0 = series0.getRawIndicesByActiveState('active');
    // console.log(indices0);
    // console.log(indices0.map(item => this.state.data[item][4]));
    this.context.dispatch({
      type: 'setSelectedUsers',
      payload: indices0.map(item => this.state.data[item][this.newKeys.findIndex(key => key === '人员编号')]), // 刷选索引映射到数据维度
    });
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

export default Parallel;