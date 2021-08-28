import React, { Component } from 'react';
import axios from 'axios';
import Charts from '../Charts';
import Bar from '../components/chart-bar/Bar';


class ChartLeft extends Component {
  constructor(props) {
    super(props);
    this.groupOneKey = ['时间无关熵', '随机熵', '真实熵', '离家距离熵', '旅行熵'];
    this.state = {
      data: null, // 所有数据
      groupOne: [], // 数据分组一
    }
  }

  // 请求数据
  getData = (url) => {
    axios.get(url).then(
      res => {
        this.setState({
          data: res.data,
        })
      }
    )
  }

  filterData = (data, x, y) => {
    let res = [];
    for (let value of Object.values(data)) {
      res.push([value[x], value[y]]);
    }
    res = res.sort((a, b) => (a[1] - b[1] > 0));
    return res
  }

  // 根据 y 轴刻度标签生成数据
  generateDataByYAxis = (arr, xAxis) => {
    return arr.map(item => (
      { title: item, data: () => this.filterData(this.state.data, xAxis, item) }
    ))
  }

  componentDidMount() {
    this.getData(`${process.env.PUBLIC_URL}/mock/ocean_score.json`); // 请求数据
  }

  componentDidUpdate(prevProps, prevState) {
    // 请求到数据后的业务逻辑
    if (prevState.data !== this.state.data) {
      this.setState({
        groupOne: this.generateDataByYAxis(this.groupOneKey, '人员编号'),
      })
    }
  }

  render() {
    return (
      <Charts.Group>
        <Charts.Box data={this.state.groupOne} dataKey={this.groupOneKey}>
          {(data) => (<Bar height='200px' yAxisName="熵值" data={data}  />)}
        </Charts.Box>
        {/* <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box> */}
      </Charts.Group>
    );
  }
}

export default ChartLeft;