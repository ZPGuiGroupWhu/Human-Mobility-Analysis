import React, { Component } from 'react';
import axios from 'axios';
import Charts from '../Charts';
import Bar from '../components/chart-bar/Bar';
import Scatter from '../components/chart-scatter/Scatter';
import Histogram from '../components/chart-histogram/Histogram';
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

  render() {
    return (
      <Charts.Group>
        {
          ({ reqSuccess }) => {
            return (
              <>
                <Charts.Box1d
                  filterable={true} // 是否启用数据递进过滤按钮
                  reqSuccess={reqSuccess} // 请求状态
                  axis={this.groupOneKey}
                >
                  {(data, props = {}) => (<Histogram height="250px" data={data} {...props} />)}
                </Charts.Box1d>
                <Charts.Box2d
                  filterable={true} // 是否启用数据递进过滤按钮
                  reqSuccess={reqSuccess} // 请求状态
                  xAxis={this.groupTwoKey[0]}
                  yAxis={this.groupTwoKey[1]}
                >
                  {(data, props = {}) => (<Scatter height='250px' data={data} {...props} />)}
                </Charts.Box2d>
              </>
            )
          }
        }

      </Charts.Group>
    );
  }
}

export default ChartLeft;