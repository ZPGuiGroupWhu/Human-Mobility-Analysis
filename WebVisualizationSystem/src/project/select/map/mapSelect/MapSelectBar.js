import React, { Component, createRef } from 'react'
import * as echarts from 'echarts'
import _ from 'lodash';
import './MapSelectWindow.scss';
// 边界数据
import regionJson from '../regionJson/Shenzhen';
// react-redux
import { connect } from 'react-redux';
import { setSelectedByMapBrush } from '@/app/slice/selectSlice';

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
      mapBrushReload: {},
      isFirst: false, // 获取数据后初次渲染 标记
    }
  }
  mapRef = createRef();

  // 内容展开与关闭
  setChartVisible = () => {
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }

  // 组织用户-top5位置数据：{id1:[{lnglat:[], count:XX}, {lnglat:[], count:XX},...], id2:[{lnglat:[], count:XX}, {lnglat:[], count:XX},...] ....}
  getUserData = (selectedUsers) => {
    //重新组织数据形式，便于后续筛选
    let data = {};
    _.forEach(this.props.UsersTopFive, function (item) {
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
          color: ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d94e5d']
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
        },
        selectedMode: true, // 选中模式
        select: {
          itemStyle: {
            areaColor: '#CFB53B'
          }
        },
        emphasis: {
          itemStyle: {
            areaColor: '#CFB53B'

          }
        },
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
              keep: '开启多选',
            }
          },
          // 清除还原 功能
          myTool1: {
            show: true,
            title: '还原',
            icon:
              "M819.199242 238.932954l136.532575 0c9.421335 0 17.066098-7.644762 17.067994-17.066098l0-136.532575-34.134092 0L938.665719 174.927029C838.326316 64.646781 701.016372 0 563.20019 0 280.88245 0 51.20019 229.682261 51.20019 512s229.682261 512 512 512c160.289736 0 308.325479-72.977903 406.118524-200.225656l-27.067616-20.78799c-91.272624 118.749781-229.445258 186.879554-379.050907 186.879554-263.509197 0-477.865908-214.356711-477.865908-477.865908S299.689097 34.134092 563.20019 34.134092c131.090991 0 262.003755 63.224764 356.406712 170.664771l-100.405764 0L819.201138 238.932954z",
            onclick: () => {
              // 清除2D地图上的选kuang
              myMap.dispatchAction({
                type: 'brush',
                areas: [], // 点击reload同时清除选择框
              });
              // 标记 用于后续清除顶层的selectedByMapBrush数组
              this.setState({
                mapBrushReload: {}
              })
            },
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
        toolbox: ['rect', 'polygon', 'keep'],
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
          return (params[2] / 50);
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
    console.log('mapBrush selected users:', payload);
    this.props.setSelectedByMapBrush(payload); //更新Map筛选出的用户集
  };

  componentDidMount() {
    this.initMap();
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    //prevProps获取到的rightWidth是0，在PageSelect页面componentDidMount获取到rightWidth值后，重新初始化
    if (!_.isEqual(prevProps.rightWidth, this.props.rightWidth)) {
      this.initMap();
    }

    // 没获取到数据
    if(this.props.LocationsReqStatus !== 'succeeded') return
    // 获取到数据，初次渲染
    if(this.props.LocationsReqStatus === 'succeeded' && !this.state.isFirst){
      let usersData = this.getUserData(this.props.selectedUsers);
      this.updateMap(usersData);
      this.setState({
        isFirst: true
      })
    }
    //只要this.props.selectedUsers中的值改变，就会在地图上重新渲染
    if (!_.isEqual(prevProps.selectedUsers, this.props.selectedUsers) && this.props.LocationsReqStatus === 'succeeded') {
      let usersData = this.getUserData(this.props.selectedUsers);
      this.updateMap(usersData);
    }
    if (prevState.mapBrushReload !== this.state.mapBrushReload) {
      // 清除顶层的selectedByMapBrush数组
      this.props.setSelectedByMapBrush([]);
    }
  }


  render() {
    return (
      <div 
        ref={this.mapRef}
        style={{
          height: '100%',
          width: '100%'
        }}
      />
    );
  }
}

const mapStateToProps = (state) => ({
  LocationsReqStatus: state.select.LocationsReqStatus,
  UsersTopFive: state.select.UsersTopFive,
  selectedUsers: state.select.selectedUsers,
})

const mapDispatchToProps = (dispatch) => ({
  setSelectedByMapBrush: (payload) => dispatch(setSelectedByMapBrush(payload)),
})

export default connect(mapStateToProps, mapDispatchToProps)(MapSelectBar);