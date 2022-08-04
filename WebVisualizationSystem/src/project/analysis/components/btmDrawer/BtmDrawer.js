import React, { Component } from 'react';
import './BtmDrawer.scss';
import { Button } from 'antd';
import { UpSquareOutlined, DownSquareOutlined } from '@ant-design/icons';
import _ from 'lodash';
import CalendarWindow from '../WeekAndHour/CalendarWindow';
import FoldContent from '../foldContent/FoldContent';


class BtmDrawer extends Component {
  constructor(props) {
    super(props);
    this.state = {
      bottomHeight: 200, // 判断button 位置
      bottomBtnType: true, // 判断button 位置 和 样式
      value: 2,
    }
  }

  componentDidUpdate(prevState) { // 隐藏展开内容
    if (!_.isEqual(prevState.bottomBtnType, this.state.bottomBtnType)) {
      ((this.state.bottomBtnType === true) ?
        document.querySelector('.fold-window').style.display = 'none' :
        document.querySelector('.fold-window').style.display = ''
      )
    }
  }

  render() {
    return (
      <div className='bottom-part'>
        <div className='button'>
          <Button
            ghost={false}
            disabled={this.props.dataloadStatus ? false : true}
            icon={this.state.bottomBtnType ?
              <UpSquareOutlined /> : <DownSquareOutlined />}
            size={'small'}
            onClick={() =>
              this.setState({
                bottomBtnType: !this.state.bottomBtnType,
                bottomHeight: (!this.state.bottomBtnType ? 200 : 382) //  根据type判断位置
              }, () => {
                this.props.setBottomStyle(this.state.bottomHeight) // 返回给主页，用于设置drawer的宽和高
              })
            }
          >{this.state.bottomBtnType ? '面板展开' : '面板收起'}</Button>
        </div>
        <div className='fold-window'>
          <FoldContent
            dataloadStatus={this.props.dataloadStatus}
            userData={this.props.userData}
            date={this.props.date}
            calendarReload={this.props.calendarReload}
            characterReload={this.props.characterReload}
            setCalendarReload={this.props.setCalendarReload}
            setCharacterReload={this.props.setCharacterReload} />
        </div>
        <div className='bottom-calendar'>
          <CalendarWindow userData={this.props.userData} isVisible={true}
            setCalendarReload={this.props.setCalendarReload} calendarReload={this.props.calendarReload} />
        </div>

      </div>
    );
  }
}

export default BtmDrawer;