import React, { Component } from 'react';
import axios from 'axios';
import Charts from '../Charts';
import Bar from '../components/chart-bar/Bar';
import Scatter from '../components/chart-scatter/Scatter';
import Store from '@/store';


class ChartLeft extends Component {
  static contextType = Store;

  constructor(props) {
    super(props);
    this.groupOneKey = ['时间无关熵', '随机熵', '真实熵', '离家距离熵', '旅行熵']; // 第一组 key 值
    this.groupTwoKey = [
      ['总出行距离', '总出行次数', 'k值', '速度均值', '速度最大值'],
      ['总出行距离', '总出行次数', 'k值', '速度均值', '速度最大值'],
    ]; // 第二组 key 值
    this.state = {
      reqSuccess: false,
      groupOne: [], // 第一组数据
      groupTwo: [], // 第二组数据
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
    const { state } = this.context;
    // 请求到数据后的业务逻辑
    if (prevState.reqSuccess !== this.state.reqSuccess) {
      this.setState({
        groupOne: this.generateDataByYAxis(state.allData, this.groupOneKey, '人员编号'),
      })
    }
  }

  render() {
    return (
      <Charts.Group>
        <Charts.Box
          data={this.state.groupOne} // 数据
          dataKey={this.groupOneKey} // 数据 key 值
          sortable={true} // 是否启用排列按钮
          filterable={true} // 是否启用数据递进过滤按钮
        >
          {(data, props = {}) => (<Bar height='250px' data={data} {...props} />)}
        </Charts.Box>
        <Charts.Box2d
          xAxis={this.groupTwoKey[0]}
          yAxis={this.groupTwoKey[1]}
        >
          {(data, props = {}) => (<Scatter height='250px' data={data} {...props} />)}
        </Charts.Box2d>
      </Charts.Group>
    );
  }
}

export default ChartLeft;