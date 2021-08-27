import React, { Component } from 'react';
import Charts from '../Charts';

class ChartRight extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  render() {
    return (
      <Charts.Group>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '600px' }}></div>
        </Charts.Box>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
      </Charts.Group>
    );
  }
}

export default ChartRight;