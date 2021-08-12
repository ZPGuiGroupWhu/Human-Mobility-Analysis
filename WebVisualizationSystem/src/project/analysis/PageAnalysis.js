import React, { Component } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// Context 对象导入
import { drawerVisibility } from '@/context/mainContext'
// 样式
import '../bmap.scss';
// 伪数据
import userData from './components/deckGL/399313.json'
// 自定义组件
import DeckGLMap from './components/deckGL/DeckGLMap';
import CalendarDrawer from './components/calendar/CalendarDrawer';
import Calendar from './components/calendar/Calendar';


/**
 * props
 * @param {number[] | string} initCenter - 初始中心经纬度 | 城市名（如‘深圳市’）
 * @param {number} initZoom - 初始缩放级别
 */
class PageAnalysis extends Component {
  constructor(props) {
    super(props);
    this.EVENTNAME = 'showTraj';
    this.state = {
      date: null,
    };
  }
  static contextType = drawerVisibility;

  getTrajCounts = (count) => {
    this.setState({
      date: count,
    })
  }

  render() {
    return (
      <>
        <DeckGLMap userData={userData} getTrajCounts={this.getTrajCounts} eventName={this.EVENTNAME} />
        <CalendarDrawer render={() => (<Calendar data={this.state.date} eventName={this.EVENTNAME} />)} height={170} />
      </>
    )
  }
}

export default PageAnalysis