import React, { Component } from 'react';
import "./PageSelect.scss";
import Map from './map/Map';
import Footer from './footer/Footer';
import ChartLeft from './charts/left/ChartLeft';
import ChartRight from './charts/right/ChartRight';
import Store from '@/store';
import _ from 'lodash';

class PageSelect extends Component {
  static contextType = Store;

  constructor(props) {
    super(props);
    this.state = {
      leftWidth: 0, // 左侧栏宽度
      rightWidth: 0, //右侧栏宽度
      footerHeight: 0, //底部内容高度
      selectedByCharts: [], // 历史筛选记录(限制更新)
      selectedByCalendar: [],
    };
  }

  // 取交集
  handleIntersection = (...params) => {
    // 若存在元素不为数组类型，则报错
    let type = params.some(item => !Array.isArray(item));
    if (type) {
      throw new Error('param should be Array Type');
    }

    let result = params.reduce((prev, cur) => {
      if (prev.length === 0) return [...cur];
      if (cur.length === 0) return [...prev];
      return Array.from(new Set(prev.filter(item => cur.includes(item))))
    }, [])

    console.log(this.context.state);
    this.context.dispatch({type: 'setSelectedUsers', payload: result});
  }

  componentDidMount() {
    this.setState({
      leftWidth: document.querySelector('.left').getBoundingClientRect().right,
      rightWidth: document.querySelector('.right').getBoundingClientRect().right
          - document.querySelector('.right').getBoundingClientRect().left,
      footerHeight: document.querySelector('.footer-bar').getBoundingClientRect().bottom
          - document.querySelector('.footer-bar').getBoundingClientRect().top,
    })
  }

  componentDidUpdate(prevProps, prevState) {
    // 监听 context
    if (
      !_.isEqual(prevState.selectedByCharts, this.context.state.selectedByCharts) ||
      !_.isEqual(prevState.selectedByCalendar, this.context.state.selectedByCalendar)
    ) {
      this.handleIntersection(this.context.state.selectedByCharts, this.context.state.selectedByCalendar);
      this.setState({
        selectedByCharts: this.context.state.selectedByCharts,
        selectedByCalendar: this.context.state.selectedByCalendar,
      })
    }
  }

  componentWillUnmount() {

  }

  render() {
    return (
      <div className="select-page-ctn">
        <div className="center">
          <Map leftWidth={this.state.leftWidth} footerHeight={this.state.footerHeight}/>
          <div className="inner">
            <div className="top-bracket"></div>
            <div className="footer-bar">
              <Footer />
            </div>
          </div>
        </div>
        <div className="left">
          <ChartLeft />
        </div>
        <div className="right">
          <ChartRight rightWidth={this.state.rightWidth}/>
        </div>
      </div>
    )
  }
}

export default PageSelect