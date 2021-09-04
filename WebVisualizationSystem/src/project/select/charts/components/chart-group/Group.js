import React, { Component } from 'react';
import "./Group.scss";
import { reqData } from '../../common/HOC/reqData';
import { pipe } from '../../common/HOC/pipe';
import FuncBar from '../func-bar/FuncBar';

class Group extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }

  render() {
    return (
      <div className="chart-group-ctn">
        <FuncBar />
        {this.props.children}
      </div>
    );
  }
}

// 组合 HOC
const compose = pipe(
  reqData(`${process.env.PUBLIC_URL}/mock/ocean_score.json`),
)

export default compose(Group);