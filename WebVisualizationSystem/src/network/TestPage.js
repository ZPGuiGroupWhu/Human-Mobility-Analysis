import React, { Component } from 'react';
import {
  getUserTraj,
  getUserODs,
} from './index';

export default class TestPage extends Component {
  getData = async (id) => {
    let traj = await getUserTraj(id);
    let data = await getUserODs();
    console.log(traj, data);
  }

  componentDidMount() {
    this.getData(399313);
  }
  render() {
    return (
      <div>
        
      </div>
    )
  }
}
