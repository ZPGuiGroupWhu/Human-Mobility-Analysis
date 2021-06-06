import React, { Component } from 'react';
import * as echarts from 'echarts';
import 'echarts/extension/bmap/bmap';
import Sider from '@/components/sider/Sider';
// 样式
import './bmap.scss';

class PageSelect extends Component {
  /**
   * props
   * @param {number[] | string} initCenter - 初始中心经纬度 | 城市名（如‘深圳市’）
   * @param {number} initZoom - 初始缩放级别
   */
  constructor(props) {
    super(props);
    this.state = {
      siderShow: false,
    };

    this.ref = React.createRef(null);
    this.ctx = this;
    this.chart = null;
    // 静态配置项
    this.option = {
      // 加载 bmap 组件
      bmap: {
        center: [120.13066322374, 30.240018034923], // 初始中心
        zoom: 12, // 初始缩放级别
        roam: true, // 是否开启拖拽缩放
        mapStyle: {}, // 自定义样式，见 http://developer.baidu.com/map/jsdevelop-11.htm
      },
      series: []
    }
    // 地图实例
    this.bmap = null;
  }

  componentDidMount() {
    // 实例化 chart
    this.chart = echarts.init(this.ref.current);
    this.chart.setOption(this.option);
    // 获取地图实例, 初始化
    this.bmap = this.chart.getModel().getComponent('bmap').getBMap();
    this.bmap.centerAndZoom(this.props.initCenter, this.props.initZoom);
    this.bmap.setMapStyleV2({
      styleId: 'f65bcb0423e47fe3d00cd5e77e943e84'
    })
  }

  render() {
    return (
      <>
        <div
          key={1}
          ref={this.ref}
          className='bmap-container'
        ></div>
        <Sider key={1} floatType='left'>PageSelect</Sider>
        <Sider key={1} floatType='right'>PageSelect</Sider>
      </>
    )
  }
}

export default PageSelect