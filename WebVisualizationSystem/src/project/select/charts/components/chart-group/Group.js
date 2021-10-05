import React, { Component } from 'react';
import "./Group.scss";
import FuncBar from '../func-bar/FuncBar';

class Group extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isReload: {}, // 是否重置
      isBrushEnd: {}, // 是否结束刷选
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
          handleReload={() => { this.setState({ isReload: {} }) }} // 触发重置
        />
        {
          this.props.render({
            isReload: this.state.isReload,
            isBrushEnd: this.state.isBrushEnd,
            handleBrushEnd: this.handleBrushEnd,
          })
        }
      </div>
    );
  }
}

export default Group;