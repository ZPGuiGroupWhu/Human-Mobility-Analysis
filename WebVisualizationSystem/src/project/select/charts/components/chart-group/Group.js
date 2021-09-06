import React, { Component } from 'react';
import "./Group.scss";
import { reqData } from './reqData';
import { pipe } from '../../common/HOC/pipe';
import FuncBar from '../func-bar/FuncBar';

class Group extends Component {
  /**
   * props
   * @param {boolean} reqSuccess - 原始数据是否请求成功
   */
  constructor(props) {
    super(props);
    this.state = {
      isReload: {}, // 是否重置
      isBrushEnd: {}, // 是否结束刷选
      connect: false, // 是否联动
    }
  }

  handleBrushEnd = () => {
    this.setState({
      isBrushEnd: {},
    })
  }

  render() {
    return (
      <div className="chart-group-ctn">
        <FuncBar
          handleReload={() => this.setState({ isReload: {} })} // 触发重置
          handleConnect={() => this.setState(prev => ({ connect: !prev.connect }))} // 改变联动状态
        />
        {
          this.props.render({
            reqSuccess: this.props.reqSuccess,
            isReload: this.state.isReload,
            connect: this.state.connect,
            isBrushEnd: this.state.isBrushEnd,
            handleBrushEnd: this.handleBrushEnd,
          })
        }
      </div>
    );
  }
}

// 组合 HOC
const compose = pipe(
  reqData(`${process.env.PUBLIC_URL}/mock/ocean_score.json`),
)

export default compose(Group);