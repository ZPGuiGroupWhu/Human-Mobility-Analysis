import React, { Component, createRef } from 'react'
import * as echarts from 'echarts'
import _ from 'lodash';
import './MapSelectBar.scss';
import '@/project/border-style.scss';
import {
  CompressOutlined,
  ExpandOutlined,
} from '@ant-design/icons';
import { Space } from 'antd';
import Hover from '../../charts/common/Hover';
//测试数据
import regionJson from '../regionJson/Shenzhen';
import userLocations from '@/project/select/charts/bottom/jsonData/userLoctionCounts';
// react-redux
import { connect } from 'react-redux';
import { setSelectedByMap } from '@/app/slice/selectSlice';

let myMap = null;
class MapSelectBar extends Component {
  // icon 通用配置
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }
  constructor(props) {
    super(props);
    this.state = {
      isVisible: true
    }
  }
  mapRef = createRef();

  // 内容展开与关闭
  setChartVisible = () =>{
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }
  
  // 组织用户-top5位置数据：{id1:[{lnglat:[], count:XX}, {lnglat:[], count:XX},...], id2:[{lnglat:[], count:XX}, {lnglat:[], count:XX},...] ....}
  getUserData = (selectedUsers) => {
    //重新组织数据形式，便于后续筛选
    let data = {};
    _.forEach(userLocations, function (item) {
      data[item.id] = item.data
    });
    //存储筛选的用户数据: [{id:xx, locations: [{lnglat:[], count:xx}, {lnglat:[], count:xx},...]},...]
    let userData = [];
    for (let i = 0; i < selectedUsers.length; i++) {
      userData.push({ id: selectedUsers[i], locations: data[selectedUsers[i].toString()] })
    }
    return userData;
  };

  //初始化地图
  initMap = () => {
    //绘制基础地图的option参数
    const option = {
      tooltip: {
        formatter: function (params) {// 说明某日出行用户数量
          return '用户编号: ' + params.value[3] + '<br/>'
            + '经度: ' + params.value[0] + '<br />'
            + '纬度: ' + params.value[1] + '<br />'
            + '出行次数: ' + params.value[2];
        },
      },
      // 鼠标悬浮的字体样式
      textStyle: {
        color: '#000',
        fontFamily: 'Microsoft Yahei',
        fontSize: 12,
        fontWeight: 'bolder'
      },
      //visualMap图例
      visualMap: {
        show: false,
        dimension: 4, // 使用第5个纬度
        min: 1, // 最小值
        max: 0, // 最大用户颜色编号，每次选取数据后会更新用户的颜色编号，
        inRange: { //颜色数组
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        },
      },
      geo: {
        map: 'Shenzhen',
        roam: true,
        aspectScale: 0.8, // 地图长宽比
        zoom: 1.2, // 地图缩放比例
        nameMap: '中国深圳2D',
        top: '15%', // 距离容器顶端位置
        scaleLimit: {
          max: 5,
          min: 1,
        },
        tooltip: {
          show: false
        }
      },
      toolbox: { // 工具类
        itemSize: 15,
        itemGap: 8,
        showTitle: true,
        right: -5,
        feature: {
          brush: {
            title: {
              rect: '矩形选择',
              polygon: '任意形状选择',
              keep: '是否多选',
              clear: '清除'
            }
          }
        },
        iconStyle: {
          color: '#fff', // icon 图形填充颜色
          borderColor: '#fff', // icon 图形描边颜色
        },
        emphasis: {
          iconStyle: {
            color: '#7cd6cf',
            borderColor: '#7cd6cf',
          }
        }
      },
      brush: { // 地图选择框
        toolbox: ['rect', 'polygon', 'keep', 'clear'],
        geoIndex: 'all',
        transformable: false, // 选择框是否可以平移
        throttleType: 'debounce', // 选择后渲染
        throttleDelay: 600, // 选择后的0.6s渲染
        inBrush: { // 选中特效
          colorAlpha: 1
        },
        outOfBrush: { // 未被选中特效
          colorAlpha: 0.8
        },
      },
      series: [{
        type: 'scatter',
        name: 'scatter2D',
        coordinateSystem: 'geo',
        data: [],
        symbolSize: function (params) {
          return (params[2] / 40);
        },
        // tooltip: {
        //   show: true
        // }
      }]
    };
    // 注册地图到组件, 初始化实例对象
    echarts.registerMap('Shenzhen', regionJson);
    myMap = echarts.init(this.mapRef.current);
    myMap.setOption(option);
    myMap.on('brushSelected', this.onBrushSelected); // 添加 brushSelected 事件
    window.onresize = myMap.resize;
  };

  //更新地图
  updateMap = (barData) => {
    // 如果没有数据则warning
    if (barData.length === 0) {
      // message.warning('No selected data !', 2);//warning
      // 多发生在点击清空按钮后，因此需要将bar图层的数据设置为空
      myMap.setOption({
        series: [{
          name: 'scatter2D',
          data: []
        }]
      });
    } else {//后续这一部分，可以用一个函数来代替，this.XXX(){retuan data}
      // 获取用户id数组
      let data = [];
      let id_num = 1; // id => num 映射，每个用户的轨迹对应一个相同值，作为visualMap的颜色表示编号。
      for (let i = 0; i < barData.length; i++) {
        for (let j = 0; j < barData[i].locations.length; j++) {
          data.push([barData[i].locations[j].lnglat[0], barData[i].locations[j].lnglat[1], barData[i].locations[j].count, barData[i].id, id_num])
        }
        id_num += 1;
      }
      myMap.setOption({
        visualMap: {
          max: id_num
        },
        series: [{
          name: 'scatter2D',
          data: data
        }]
      });
    }
  };

  // 存储刷选的数据索引映射
  onBrushSelected = (params) => {
    let brushComponent = params.batch[0];
    if (!brushComponent.selected[0].dataIndex.length) return; // 若开启过滤，则始终保留历史刷选数据
    let usersData = this.getUserData(this.props.selectedUsers);
    // 构建当前绘制scatter的数据，用于寻找对应index的位置点
    let data = [];
    for (let i = 0; i < usersData.length; i++) {
      for (let j = 0; j < usersData[i].locations.length; j++) {
        data.push([usersData[i].locations[j].lnglat[0], usersData[i].locations[j].lnglat[1], usersData[i].locations[j].count, usersData[i].id])
      }
    }
    // 通过index映射为用户编号
    const payload = Array.from(new Set(brushComponent.selected[0].dataIndex.map(
      item => { return data[item][3] })));
    console.log('map selected users:', payload);
    this.props.setSelectedByMap(payload); //更新Map筛选出的用户集
  };

  componentDidMount() {
    this.initMap();
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    //prevProps获取到的rightWidth是0，在PageSelect页面componentDidMount获取到rightWidth值后，重新初始化
    if (!_.isEqual(prevProps.rightWidth, this.props.rightWidth)) {
      this.initMap();
    }
    //只要this.props.selectedUsers中的值改变，就会在地图上重新渲染
    if (!_.isEqual(prevProps.selectedUsers, this.props.selectedUsers)) {
      let usersData = this.getUserData(this.props.selectedUsers);
      this.updateMap(usersData);
    }
    if (prevProps.mapReload !== this.props.mapReload) {
      this.props.setSelectedByMap([]);
      myMap.dispatchAction({
        type: 'brush',
        areas: [], // 点击reload同时清除选择框
      })
    }
  }


  render() {
    return (
      <div
        className="map-select-bar tech-border"
        style={{
          right: this.props.right + 5,
          bottom: this.props.bottom + 5
        }}>
        <div className='title-bar'>
          <div className='map-box-title'>
            <span style={{ color: '#fff', fontFamily: 'sans-serif', fontSize: '15px', fontWeight: 'bold' }}>{'中国深圳2D'}</span>
          </div>
          <div className='map-box-switch'>
            <Space>
              {
                this.state.isVisible ?
                  <Hover>
                    {
                      ({ isHovering }) => (
                        <CompressOutlined
                          style={{
                            ...this.iconStyle,
                            color: isHovering ? '#05f8d6' : '#fff'
                          }}
                          onClick={this.setChartVisible}
                        />
                      )
                    }
                  </Hover>
                  :
                  <Hover>
                    {
                      ({ isHovering }) => (
                        <ExpandOutlined
                          style={{
                            ...this.iconStyle,
                            color: isHovering ? '#05f8d6' : '#fff'
                          }}
                          onClick={this.setChartVisible}
                        />
                      )
                    }
                  </Hover>
              }
            </Space>
          </div>
        </div>
        <div className='map-select-function' ref={this.mapRef}>
        </div>
      </div>

    );
  }
}

const mapStateToProps = (state) => ({
  selectedUsers: state.select.selectedUsers,
})

const mapDispatchToProps = (dispatch) => ({
  setSelectedByMap: (payload) => dispatch(setSelectedByMap(payload)),
})

export default connect(mapStateToProps, mapDispatchToProps)(MapSelectBar);