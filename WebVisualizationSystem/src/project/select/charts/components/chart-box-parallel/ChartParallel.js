import React, { Component } from 'react';
import { CSSTransition } from 'react-transition-group';
// HOC
import { filterBySelect } from '../../common/HOC/filterBySelect';
import { pipe } from '../../common/HOC/pipe';

class ChartParallel extends Component {
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
        classNames='chart-parallel'
        onEnter={(node) => { node.style.setProperty('display', '') }}
        onExiting={(node) => { node.style.setProperty('display', 'none') }}
      >
        <div className="chart-parallel-content">
          {
            this.props.render({
              data: this.props.data,
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
)

export default compose(ChartParallel);