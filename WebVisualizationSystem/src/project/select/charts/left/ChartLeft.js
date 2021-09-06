import React, { Component } from 'react';
import Charts from '../Charts';
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
      <Charts.Group
        render={
          ({ reqSuccess, isReload, connect, isBrushEnd, handleBrushEnd }) => (
            <>
              <Charts.Box
                id={1}
                reqSuccess={reqSuccess}
                isReload={isReload}
                connect={connect}
                xAxis={this.groupOneKey}
                yAxis={'人数'}
                isBrushEnd={isBrushEnd}
                handleBrushEnd={handleBrushEnd}
                render={
                  ({ data, xAxisName, yAxisName, handleBrushEnd }) => (
                    <Histogram
                      height="250px"
                      data={data}
                      xAxisName={xAxisName}
                      yAxisName={yAxisName}
                      handleBrushEnd={handleBrushEnd}
                    />
                  )
                }
              >
              </Charts.Box>
              <Charts.Box
                id={2}
                reqSuccess={reqSuccess}
                isReload={isReload}
                connect={connect}
                xAxis={this.groupTwoKey[0]}
                yAxis={this.groupTwoKey[1]}
                isBrushEnd={isBrushEnd}
                handleBrushEnd={handleBrushEnd}
                render={
                  ({ data, xAxisName, yAxisName, handleBrushEnd }) => (
                    <Scatter
                      height="250px"
                      data={data}
                      xAxisName={xAxisName}
                      yAxisName={yAxisName}
                      handleBrushEnd={handleBrushEnd}
                    />
                  )
                }
              >
              </Charts.Box>
            </>
          )
        }
      ></Charts.Group>
    );
  }
}

export default ChartLeft;