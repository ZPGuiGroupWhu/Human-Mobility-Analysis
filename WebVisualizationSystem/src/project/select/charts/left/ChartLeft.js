import React, { Component } from 'react';
import axios from 'axios';
import Charts from '../Charts';
import Bar from '../components/chart-bar/Bar';
import Store from '@/store';


class ChartLeft extends Component {
  static contextType = Store;

  constructor(props) {
    super(props);
    this.groupOneKey = ['时间无关熵', '随机熵', '真实熵', '离家距离熵', '旅行熵']; // 第一组 key 值
    this.state = {
      reqSuccess: false,
      groupOne: [], // 第一组数据
    }
  }

  // 请求数据
  getData = (url) => {
    axios.get(url).then(
      res => {
        this.context.dispatch({ type: 'setAllData', payload: res.data });
        this.setState({ reqSuccess: true }); // 数据请求成功
      }
    )
  }

  // 数据过滤
  filterData = (data, x, y) => {
    let res = [];
    for (let value of Object.values(data)) {
      res.push([value[x], value[y]]);
    }
    return res
  }

  // 根据 y 轴刻度标签生成数据
  generateDataByYAxis = (data, keys, xAxis) => {
    try {
      if (!Array.isArray(keys)) throw new Error('keys should be Array Type');
      if (!data) throw new Error('data should not be null');
      return keys.map(item => (
        { title: item, data: () => this.filterData(data, xAxis, item) }
      ))
    } catch (err) {
      console.log(err);
    }  
  }

  componentDidMount() {
    this.getData(`${process.env.PUBLIC_URL}/mock/ocean_score.json`); // 请求数据
  }

  componentDidUpdate(prevProps, prevState) {
    const {state} = this.context;
    // 请求到数据后的业务逻辑
    if (prevState.reqSuccess !== this.state.reqSuccess) {
      this.setState({
        groupOne: this.generateDataByYAxis(state.allData, this.groupOneKey, '人员编号'),
      })
    }
    console.log(state);
  }

  render() {
    return (
      <Charts.Group>
        <Charts.Box data={this.state.groupOne} dataKey={this.groupOneKey}>
          {(data) => (<Bar height='250px' yAxisName="熵值" data={data} />)}
        </Charts.Box>
      </Charts.Group>
    );
  }
}

export default ChartLeft;