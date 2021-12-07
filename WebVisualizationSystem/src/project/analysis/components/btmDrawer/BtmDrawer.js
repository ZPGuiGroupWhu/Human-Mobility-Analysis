import React, { Component } from 'react';
import { Radio, Space } from 'antd';
import BottomCalendar from '../calendar/BottomCalendar';
import TimerLine from '../timePlayer/TimerLine';
import './BtmDrawer.scss';

class BtmDrawer extends Component {
  constructor(props) {
    super(props);
    this.radioContents = [
      { id: 1, text: '日历'},
      { id: 2, text: '时间窗口'},
      { id: 3, text: '时间轴'},
    ]
    this.state = {
      value: 1,
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
        {this.state.value === 1 && this.props.date ? <BottomCalendar data={this.props.date} eventName={this.props.EVENTNAME} /> : null}
        {this.state.value === 2 && null}
        {this.state.value === 3 && <TimerLine />}
      </>
    );
  }
}

export default BtmDrawer;