import React, { Component } from 'react';
import { Radio, Space } from 'antd';
import BottomCalendar from '../calendar/BottomCalendar';
import './BtmDrawer.scss';
import _ from 'lodash';
import CalendarWindow from '../WeekAndHour/CalendarWindow';

class BtmDrawer extends Component {
  constructor(props) {
    super(props);
    this.radioContents = [
      { id: 1, text: '日历' },
      { id: 2, text: '时间窗口' },
    ];
    this.state = {
      value: 2,
    }
  }

  onChange = (e) => {
    this.setState({
      value: e.target.value,
    })
  }



  render() {
    return (
      <>
        <div className='btmdrw-radio btmdrw-moudle'>
          <Radio.Group onChange={this.onChange} value={this.state.value}>
            <Space direction="vertical">
              {this.radioContents.map(item => (<Radio value={item.id}>{item.text}</Radio>))}
            </Space>
          </Radio.Group>
        </div>
        {
          (this.state.value === 1 && this.props.dataloadStatus && Object.keys(this.props.date).length) ?
            <BottomCalendar data={this.props.date} eventName={this.props.EVENTNAME} /> : null
        }
        {this.state.value === 2 && <CalendarWindow />}
      </>
    );
  }
}

export default BtmDrawer;