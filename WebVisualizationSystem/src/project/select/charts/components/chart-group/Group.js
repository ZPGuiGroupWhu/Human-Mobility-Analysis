import React, { Component } from 'react';
import "./Group.scss";

class Group extends Component {
  constructor(props) {
    super(props);
    this.state = {
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
        {
          this.props.render({
            isBrushEnd: this.state.isBrushEnd,
            handleBrushEnd: this.handleBrushEnd,
          })
        }
      </div>
    );
  }
}

export default Group;