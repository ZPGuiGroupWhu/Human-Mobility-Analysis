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
        }
      },
      series: [
        {
          name: 'POI分布',
          type: 'pie',
          radius: ['60%', '80%'], // Array<number|string> - [内半径，外半径]
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
              fontSize: '20',
              fontWeight: 'bold'
            }
          },
          labelLine: {
            show: false
          },
          data: []
        }
      ]
    };
    this.state = {};
  }

  componentDidMount() {
    this.myChart.current = echarts.init(this.ref.current);
    this.myChart.current.setOption(this.option);
  }

  componentDidUpdate(prevProps, prevState) {
    if (!_.isEqual(this.props.data, prevProps.data)) {
      if (typeof this.props.data === 'object') {
        this.option.series[0].data = this.props.data;
        this.myChart.current.setOption(this.option);
      }
    }
  }

  componentWillUnmount() {
    this.myChart.current.dispose();
  }

  render() {
    return (
      <div
        ref={this.ref}
        style={{
          width: '350px',
          height: '200px',
          ...this.props.style,
        }}
      ></div>
    );
  }
}

export default Doughnut;