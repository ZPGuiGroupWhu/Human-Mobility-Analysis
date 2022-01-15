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
      isVisible: [
        false,
        true,
        false
      ]
    }
  }

  onChange = (e) => {
    this.setState((prev) => {
      return {
        value: e.target.value,
        isVisible: prev.isVisible.map((item, index) => { return index === e.target.value - 1 ? true : false })
      }
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
          (this.props.dataloadStatus && Object.keys(this.props.date).length) ?
            <BottomCalendar timeData={this.props.date} userData={this.props.userData} eventName={this.props.EVENTNAME} 
            isVisible={this.state.isVisible[0]} clear={this.props.calendarReload}/> : null
        }
        <CalendarWindow userData={this.props.userData} isVisible={this.state.isVisible[1]} clear={this.props.calendarReload}/>
        {(this.props.dataloadStatus && Object.keys(this.props.date).length) ?
          <CharacterWindow userData={this.props.userData} isVisible={this.state.isVisible[2]} clear={this.props.characterReload}/> : null
        }
      </>
    );
  }
}

export default BtmDrawer;