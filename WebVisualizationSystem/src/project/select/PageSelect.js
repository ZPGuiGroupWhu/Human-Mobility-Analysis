import React, { Component } from 'react';
import "./PageSelect.scss";
import Map from './map/Map';
import Footer from './footer/Footer';
import ChartLeft from './charts/left/ChartLeft';
import ChartBottom from './charts/bottom/ChartBottom';
import Store from '@/store';
import _ from 'lodash';

class PageSelect extends Component {
  static contextType = Store;

  constructor(props) {
    super(props);
    this.state = {
      leftWidth: 0, // 左侧栏宽度
      bottomHeight: 0, //底部内容高度
      bottomWidth: 0, //底部内容宽度
      selectedByCharts: [], // 历史筛选记录(限制更新)
      selectedByCalendar: [], //历史---日历筛选记录
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
    }, []);

    console.log(this.context.state);
    this.context.dispatch({type: 'setSelectedUsers', payload: result});
  };

  componentDidMount() {
    //返回各组件的边界位置，用于日历、map等组件的布局
    const leftWidth = document.querySelector('.left').getBoundingClientRect().right;
    const bottomHeight = document.querySelector('.bottom').getBoundingClientRect().bottom
        - document.querySelector('.bottom').getBoundingClientRect().top;
    const bottomWidth = document.querySelector('.bottom').getBoundingClientRect().right
        - document.querySelector('.bottom').getBoundingClientRect().left;
    this.setState({
      leftWidth: leftWidth,
      bottomHeight: bottomHeight,
      bottomWidth: bottomWidth,
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
      });
    }
  }

  componentWillUnmount() {

  }

  render() {
    return (
      <div className="select-page-ctn">
        <div className="center">
          <Map leftWidth={this.state.leftWidth} bottomHeight={this.state.bottomHeight}/>
          <div className="inner">
            <div className="top-bracket"></div>
            <div className="bottom">
               <ChartBottom bottomHeight={this.state.bottomHeight} bottomWidth={this.state.bottomWidth}/>
            </div>
          </div>
        </div>
        <div className="left">
          <ChartLeft />
        </div>
        <div className="footer-bar" style={{float:"right"}} >
          <Footer selectedByCharts={this.state.selectedByCharts} selectedByCalendar={this.state.selectedByCalendar}/>
        </div>
      </div>
    )
  }
}

export default PageSelect