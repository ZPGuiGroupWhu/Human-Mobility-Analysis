import React, { Component } from 'react';
import Charts from '../Charts';

class ChartLeft extends Component {
  constructor(props) {
    super(props);
    this.state = {}
  }
  render() {
    return (
      <Charts.Group>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
        <Charts.Box title="测试">
          <div style={{ backgroundColor: '#fff', height: '200px' }}></div>
        </Charts.Box>
      </Charts.Group>
    );
  }
}

export default ChartLeft;