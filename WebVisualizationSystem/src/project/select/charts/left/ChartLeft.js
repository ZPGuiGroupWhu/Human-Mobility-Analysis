import React, { Component } from 'react';
import Charts from '../Charts';
import Scatter from '../components/chart-scatter/Scatter';
import Histogram from '../components/chart-histogram/Histogram';
import Parallel from '../components/chart-parallel/Parallel';


class ChartLeft extends Component {

  constructor(props) {
    super(props);
    this.keys = ['总出行次数', '总出行距离', '旋转半径', 'k值', '时间无关熵', '旅行熵', '日内节律熵', '速度均值', '速度标准差均值']
    this.groupOneKey = this.keys; // 第一组 key 值
    this.groupTwoKey = [this.keys, this.keys]; // 第二组 key 值
    this.groupThreeKey = ['外向性', '开放性', '神经质性', '尽责性']; // 第三组 key 值
    this.state = {}
  }

  render() {
    return (
      <Charts.Group
        render={
          ({ isBrushEnd, handleBrushEnd }) => (
            <>
              <Charts.Box
                id={1}
                isReload={this.props.chartsReload}
                xAxis={this.groupOneKey}
                yAxis={'人数'}
                isBrushEnd={isBrushEnd}
                handleBrushEnd={handleBrushEnd}
                render={
                  ({ data, xAxisName, yAxisName, handleBrushEnd, isReload }) => (
                    !this.props.width ? null : // width 为 0 时，无法绘制，因此要监听 width 变化
                      <Histogram
                        height="250px"
                        data={data}
                        isReload={isReload}
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
                isReload={this.props.chartsReload}
                xAxis={this.groupTwoKey[0]}
                yAxis={this.groupTwoKey[1]}
                isBrushEnd={isBrushEnd}
                handleBrushEnd={handleBrushEnd}
                render={
                  ({ data, xAxisName, yAxisName, handleBrushEnd, isReload }) => (
                    !this.props.width ? null :
                      <Scatter
                        height="250px"
                        data={data}
                        isReload={isReload}
                        xAxisName={xAxisName}
                        yAxisName={yAxisName}
                        handleBrushEnd={handleBrushEnd}
                      />
                  )
                }
              >
              </Charts.Box>
              <Charts.BoxParallel
                id={3}
                isReload={this.props.chartsReload}
                isBrushEnd={isBrushEnd}
                handleBrushEnd={handleBrushEnd}
                title="大五人格"
                render={
                  ({ data, handleBrushEnd }) => (
                    !this.props.width ? null :
                      <Parallel
                        height="250px"
                        data={data}
                        keys={this.groupThreeKey}
                        handleBrushEnd={handleBrushEnd}
                      />
                  )
                }
              >
              </Charts.BoxParallel>
            </>
          )
        }
      ></Charts.Group>
    );
  }
}

export default ChartLeft;