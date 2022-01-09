import React, { Component } from 'react';
import { Radio, Space } from 'antd';
import BottomCalendar from '../calendar/BottomCalendar';
import './BtmDrawer.scss';
import _ from 'lodash';
import CalendarWindow from '../WeekAndHour/CalendarWindow';
import CharacterWindow from '../characterSelect/CharacterWindow';

class BtmDrawer extends Component {
  constructor(props) {
    super(props);
    this.radioContents = [
      { id: 1, text: '日历筛选' },
      { id: 2, text: '星期筛选' },
      { id: 3, text: '特征筛选' }
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
            <BottomCalendar timeData={this.props.date} userData = {this.props.userData} eventName={this.props.EVENTNAME} /> : null
        }
        {this.state.value === 2 && <CalendarWindow userData = {this.props.userData}/>}
        {this.state.value === 3 && <CharacterWindow userData = {this.props.userData}/>}
      </>
    );
  }
}

export default BtmDrawer;