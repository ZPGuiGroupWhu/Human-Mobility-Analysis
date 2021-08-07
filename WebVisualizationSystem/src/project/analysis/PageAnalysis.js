import React, { Component } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
// Context 对象导入
import { drawerVisibility } from '@/context/mainContext'
// 样式
import '../bmap.scss';
import DeckGLMap from './components/deckGL/DeckGLMap'
import userData from './components/deckGL/399313.json'


class PageAnalysis extends Component {
  /**
   * props
   * @param {number[] | string} initCenter - 初始中心经纬度 | 城市名（如‘深圳市’）
   * @param {number} initZoom - 初始缩放级别
   */
  constructor(props) {
    super(props);
    this.state = {};
  }

  static contextType = drawerVisibility;

  render() {
    return (
      <>
        <DeckGLMap userData={userData} />
      </>
    )
  }
}

export default PageAnalysis