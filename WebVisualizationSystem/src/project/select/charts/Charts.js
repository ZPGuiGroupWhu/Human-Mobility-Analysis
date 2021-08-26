import React, { Component } from 'react';
import "./Charts.scss";

class Charts extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  render() {
    return (
      <div className="select-charts-ctn">
        {this.props.children}
      </div>
    );
  }
}

export default Charts;