import React, { Component } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';

class Doughnut extends Component {
  constructor(props) {
    super(props);
    this.ref = React.createRef(); // div-dom
    this.myChart = React.createRef(); // echarts-instance
    this.option = {
      // 提示框
      tooltip: {
        trigger: 'item'
      },
      // 图例
      legend: {
        orient: 'vertical',
        left: 'right',
        top: 'middle',
        textStyle: {
          color: '#fff',
        },
        itemGap: 5,
        align: 'left',
      },
      series: [
        {
          name: 'POI分布',
          type: 'pie',
          radius: ['70%', '90%'], // Array<number|string> - [内半径，外半径]
          center: ['35%', '50%'],
          avoidLabelOverlap: false,
          label: {
            show: false,
            position: 'center',
            color: '#fff',
            fontSize: 10,
          },
          emphasis: {
            label: {
              show: true,
              fontSize: '18',
              fontWeight: 'bold',
              formatter: '{b}:{c}',
            }
          },
          labelLine: {
            show: false
          },
          data: []
        }
      ]
    };
    this.timer = null; // 计时器
    this.curIdx = 0; // 当前索引
    this.prevIdx = -1; // 历史索引
    this.state = {
      curIdx: 0,
      prevIdx: -1,
    };
  }

  setCarousel = (data) => {
    const lens = data.length;
    const carousel = () => {
      if (this.curIdx >= lens) { this.curIdx = 0 };
      if (data[this.curIdx].value === 0) {
        this.curIdx += 1;
        carousel.call(this);
      } else {
        this.myChart.current.dispatchAction({
          type: 'downplay',
          seriesName: 'POI分布',
          dataIndex: this.prevIdx,
        });
        this.myChart.current.dispatchAction({
          type: 'highlight',
          seriesName: 'POI分布',
          dataIndex: this.curIdx,
        });
        this.prevIdx = this.curIdx; // 记录上一次高亮索引
        this.curIdx += 1; // 当前索引后移
      }
    }
    // 仅当有索引结果时执行
    if (data.some(item => item.value !== 0)) {
      if (this.timer) {
        clearInterval(this.timer);
      }
      this.timer = setInterval(() => {
        carousel.call(this);
      }, this.props.autoplayInterval)
    }
  }

  onHighlight = (params) => {
    // 清除轮播
    if (this.timer) {
      clearInterval(this.timer);
    }
    // 清除旧的高亮样式
    this.myChart.current.dispatchAction({
      type: 'downplay',
      seriesName: 'POI分布',
    });
    // 手动高亮
    this.myChart.current.dispatchAction({
      type: 'highlight',
      seriesName: 'POI分布',
      dataIndex: params.dataIndex,
    });
  }

  onDownplay = () => {
    // 清除旧的高亮样式
    this.myChart.current.dispatchAction({
      type: 'downplay',
      seriesName: 'POI分布',
    });
    // 重新设置轮播
    this.setCarousel(this.props.data);
  }

  componentDidMount() {
    this.myChart.current = echarts.init(this.ref.current);
    this.myChart.current.setOption(this.option);
    this.myChart.current.on('mouseover', { seriesName: 'POI分布' }, this.onHighlight); // 绑定 mouseover 事件
  }

  componentDidUpdate(prevProps, prevState) {
    if (!_.isEqual(this.props.data, prevProps.data)) {
      console.log(this.props.data, prevProps.data);
      if (typeof this.props.data === 'object') {
        this.option.series[0].data = this.props.data;
        this.myChart.current.setOption(this.option);
        this.props.autoplay && this.setCarousel(this.props.data); // 自动高亮
        this.myChart.current.off('mouseout', this.onDownplay); // 清除旧 mouseout 事件
        this.myChart.current.on('mouseout', { seriesName: 'POI分布' }, this.onDownplay); // 绑定新 mouseout 事件
        setTimeout(() => {
          this.myChart.current.resize();
        }, 0)
      }
    }
  }

  componentWillUnmount() {
    if (this.timer) {
      clearInterval(this.timer);
    }
    this.myChart.current.off('mouseout', this.onDownplay); // 清除 mouseout 事件
    this.myChart.current.off('mouseover', this.onHighlight); // 清除 mouseover 事件
    this.myChart.current.dispose();

  }

  render() {
    return (
      <div
        ref={this.ref}
        style={{
          width: '260px',
          height: '160px',
          ...this.props.style,
        }}
      ></div>
    );
  }
}

export default Doughnut;