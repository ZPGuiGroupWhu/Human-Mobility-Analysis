import React, { Component } from 'react';
import "./Group.scss";

class Group extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  render() {
    return (
      <div className="chart-group-ctn">
          {this.props.children}
      </div>
    );
  }
}

export default Group;