import React, { Component } from 'react';
import { CSSTransition } from 'react-transition-group';
// HOC
import { filterBySelect } from '../../common/HOC/filterBySelect';
import { filterByAxis } from '../../common/HOC/filterByAxis';
import { pipe } from '../../common/HOC/pipe';

class Chart extends Component {

  /**
   * props
   * @param {boolean} isVisible - 是否可视
   * @param {object} data - 数据源
   * @param {string} xAxis - x 轴
   * @param {string} yAxis - y 轴
   */
  constructor(props) {
    super(props);
    this.state = {}
  }

  render() {
    return (
      <CSSTransition
        in={this.props.isVisible}
        timeout={300}
        classNames='chart'
        onEnter={(node) => { node.style.setProperty('display', '') }}
        onExiting={(node) => { node.style.setProperty('display', 'none') }}
      >
        <div className="chart-content">
          {
            this.props.render({
              data: this.props.data,
              isReload: this.props.isReload,
              xAxisName: this.props.xAxis,
              yAxisName: this.props.yAxis,
              handleBrushEnd: this.props.handleBrushEnd,
            })
          }
        </div>
      </CSSTransition>
    );
  }
}

const compose = pipe(
  filterBySelect(),
  filterByAxis(),
)

export default compose(Chart);