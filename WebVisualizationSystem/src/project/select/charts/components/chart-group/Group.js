import React, { Component } from 'react';
import "./Group.scss";
import { reqData } from '../../common/HOC/reqData';
import { pipe } from '../../common/HOC/pipe';
import FuncBar from '../func-bar/FuncBar';

class Group extends Component {
  /**
   * props
   * @param {boolean} reqSuccess - 原始数据是否请求成功
   */
  constructor(props) {
    super(props);
    this.state = {}
  }

  render() {
    return (
      <div className="chart-group-ctn">
        <FuncBar />
        {this.props.children({ reqSuccess: this.props.reqSuccess })}
      </div>
    );
  }
}

// 组合 HOC
const compose = pipe(
  reqData(`${process.env.PUBLIC_URL}/mock/ocean_score.json`),
)

export default compose(Group);