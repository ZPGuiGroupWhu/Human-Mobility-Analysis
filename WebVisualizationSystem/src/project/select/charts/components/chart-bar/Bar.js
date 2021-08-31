import React, { Component } from 'react';
import * as echarts from 'echarts';
import Store from '@/store';

class Bar extends Component {
  static contextType = Store; // 挂载全局状态管理

  constructor(props) {
    super(props);
    // state
    this.state = {
      selectedData: [], // 筛选出来用于视图更新的数据
    }
  }

  ref = React.createRef(null);
  chart = null;

  option = {
    tooltip: {
      show: true,
      trigger: 'axis',
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
      toolbox: ['rect', 'lineX', 'lineY', 'keep', 'clear'],
      xAxisIndex: 0
    },
    // grid - 定位图表在容器中的位置
    grid: {
      show: true, // 是否显示直角坐标系网格
      left: '40', // 距离容器左侧距离
      top: '30', // 距离容器上侧距离
      right: '40',
      bottom: '40',
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
      nameLocation: 'end', // 坐标轴名称显示位置
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
        formatter: (value, index) => (value.toFixed(3)),
      }
    },
    dataZoom: [
      {
        type: 'slider',
        xAxisIndex: 0,
        filterMode: 'filter', // 过滤模式
        textStyle: {
          color: '#fff',
          fontSize: 12,
        },
        left: 37,
        right: 43,
        // --- 百分比 ---
        // start: 0, // 初始起始范围
        // end: 4, // 初始结束范围
        // minSpan: 10, // 最小取值窗口(%)
        // maxSpan: 20, // 最大取值窗口(%)
        // --- 实值 ---
        startValue: 0,
        endValue: 20,
        minValueSpan: 6,
        maxValueSpan: 50,
        // -----------
        orient: 'horizontal', // 布局方式
        zoomLock: false, // 是否锁定取值窗口(若锁定，只能平移不能缩放)
      },
      {
        type: 'slider',
        yAxisIndex: 0,
        filterMode: 'filter', // 过滤模式
        textStyle: {
          color: '#fff',
          fontSize: 12,
        },
        top: 28,
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
    ]
  };

  // 获取筛选数据
  getSelectedData = (selectedId) => {
    if (!Array.isArray(selectedId)) throw new Error('SelectedId should be Array Type');
    return selectedId.map(val => (this.props.data.find(item => (item[0] === val))))
  }

  // 数据驱动更新视图
  reSetOption = (data) => {
    this.option.series[0].data = data
    this.chart.setOption(this.option)
  }

  // 存储刷选的数据索引映射
  onBrushSelected = (params) => {
    let brushComponent = params.batch[0];

    if (this.props.withFilter && !brushComponent.selected[0].dataIndex.length) return; // 若开启过滤，则始终保留历史刷选数据
    this.context.dispatch({
      type: 'setSelectedUsers',
      payload: brushComponent.selected[0].dataIndex.map(item => this.props.data[item][0]), // 刷选索引映射到数据维度
    });
  }

  componentDidMount() {
    this.chart = echarts.init(this.ref.current); // 初始化容器
    this.chart.on('brushSelected', this.onBrushSelected); // 添加 brushSelected 事件
    this.chart.setOption(this.option); // 初始化视图
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevProps.data !== this.props.data) {
      // 判断是否开启过滤模式
      if (this.props.withFilter) {
        // 若开启，则当存在选中用户时，切换类型加载选中数据；若不开启，或当前没有选中用户时，加载数据源
        const data = this.context.state.selectedUsers;
        const res = this.getSelectedData(data);
        this.reSetOption(
          !!data.length ? res : this.props.data
        );
        this.setState({
          selectedData: res,
        })
        this.chart.dispatchAction({ type: 'brush', areas: [] }); // 清除框选
      } else {
        this.reSetOption(this.props.data);
      }
    }


    if (prevProps.sortedData !== this.props.sortedData) {
      this.reSetOption(this.props.sortedData); // 根据排序更新视图
      this.chart.dispatchAction({ type: 'brush', areas: [] }); // 清除框选
    }

    // 触发一次排序
    if (prevProps.isSorted !== this.props.isSorted) {
      // 若开启过滤功能，则排序筛选的数据，反之，排序所有数据
      if (this.props.withFilter) {
        this.props.setSortableData(this.state.selectedData, 1);
      } else {
        this.props.setSortableData(this.props.data, 1);
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

export default Bar;