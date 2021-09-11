import React, { Component } from 'react';
import Charts from '../Charts';
import Calendar from './components/Calendar';
import { Button } from "antd";
import { RedoOutlined } from "@ant-design/icons";
import { eventEmitter } from "@/common/func/EventEmitter";
import _ from 'lodash'
//数据
import dateCounts from './jsonData/date_counts'

class ChartRight extends Component {
  constructor(props) {
    super(props);
    this.data = {};
    this.state = {
    }
  }

  // 组织日历数据：{ 日期date：出行用户数量count }
  getAllUsersData = () => {
    let data = {};
    _.forEach(dateCounts, function (item) {
      data[item.date] = { 'count': item.counts, 'users': item.users }
    });
    this.data = data;
  };

  // 加载组件前传递数据给calendar
  componentWillMount() {
    this.getAllUsersData();
  }

  componentDidMount() {
  }

  render() {
    return (
      <>
        <div>
          <Button
            ghost
            size='small'
            type='default'
            icon={<RedoOutlined style={{ color: '#fff' }} />}
            onClick={(e) => {
              let clear = true;
              eventEmitter.emit('clearCalendarHighlight', { clear })
            }}
            style={{
              position: 'absolute',
              right: '10px',
              zIndex: '9999' //至于顶层
            }}
          /></div>
        <Calendar data={this.data} />
      </>
    );
  }
}

export default ChartRight;