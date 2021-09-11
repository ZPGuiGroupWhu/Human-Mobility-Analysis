import React, { Component } from 'react';
import Charts from '../Charts';
import Scatter from '../components/chart-scatter/Scatter';
import Histogram from '../components/chart-histogram/Histogram';
import Store from '@/store';


class ChartLeft extends Component {
  static contextType = Store;

  constructor(props) {
    super(props);
    this.keys = ['总出行次数', '总出行距离', '旋转半径', 'k值', '时间无关熵', '旅行熵', '日内节律熵', '速度均值', '速度标准差均值']
    this.groupOneKey = this.keys; // 第一组 key 值
    this.groupTwoKey = [this.keys, this.keys]; // 第二组 key 值
    this.state = {}
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
              <Charts.Box
                id={3}
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
            </>
          )
        }
      ></Charts.Group>
    );
  }
}

export default ChartLeft;