import React, { Component } from 'react';
import { Radio, Space } from 'antd';
import BottomCalendar from '../calendar/BottomCalendar';
import WeekHourCalendar from '../WeekAndHour/WeekHourCalendar';
import TimerLine from '../timePlayer/TimerLine';
import ShoppingCart from '../shopping/ShoppingCart';
import './BtmDrawer.scss';
import axios from 'axios';
import _ from 'lodash';
import CalendarWindow from '../WeekAndHour/CalendarWindow';

class BtmDrawer extends Component {
  constructor(props) {
    super(props);
    this.radioContents = [
      { id: 1, text: '日历' },
      { id: 2, text: '时间窗口' },
      { id: 3, text: '购物车' },
    ];
    this.state = {
      value: 1,
      ShenZhen: null, // 深圳json边界
    }
  }

  onChange = (e) => {
    this.setState({
      value: e.target.value,
    })
  }

  componentDidMount() {
    // 深圳 json 数据
    axios.get(process.env.PUBLIC_URL + '/ShenZhen.json').then(data => {
      this.setState({ ShenZhen: data.data })
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
        {this.state.value === 2 && <CalendarWindow />}
        <ShoppingCart
          ShenZhen={this.state.ShenZhen}
          isSelected={this.state.value === 3}
        />
      </>
    );
  }
}

export default BtmDrawer;