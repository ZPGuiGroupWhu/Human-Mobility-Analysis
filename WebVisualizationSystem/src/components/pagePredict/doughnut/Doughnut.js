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
        top: '16%',
        right: '3%',
        bottom: '2%',
        left: 'center',
        textStyle: {
          color: '#fff',
        }
      },
      series: [
        {
          name: 'POI分布',
          type: 'pie',
          center: ['20%', '50%'],
          radius: ['40%', '60%'], // Array<number|string> - [内半径，外半径]
          avoidLabelOverlap: false,
          // itemStyle: {
          //   borderRadius: 10,
          //   borderColor: '#fff',
          //   borderWidth: 2
          // },
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
      this.option.series[0].data = this.props.data;
      this.myChart.current.setOption(this.option);
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
          width: '500px',
          height: '250px',
          ...this.props.style,
        }}
      ></div>
    );
  }
}

export default Doughnut;