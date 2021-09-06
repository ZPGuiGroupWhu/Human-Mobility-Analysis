import React, { Component } from 'react';
import Charts from '../Charts';
import Calendar from './components/Calendar';
import {Button} from "antd";
import {eventEmitter} from "@/common/func/EventEmitter";

class ChartRight extends Component {
  constructor(props) {
    super(props);
    this.state = {
    }
  }

  // 组织日历数据：{ 日期date：出行用户数量count }
  getAllUsersData = () => {
  };

  componentDidMount() {
    this.getAllUsersData();
  }

  render() {
    return (
      <>
        <Button
          type="primary"
          onClick={(e) => {
            let clear = true;
            eventEmitter.emit('clearCalendarHighlight', {clear})
          }}>Clear</Button>
        <Calendar data={{'2018-01-01':{count:100}}}/>
      </>
    );
  }
}

export default ChartRight;