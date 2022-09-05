import React, { Component } from 'react';
// style
import "./PageSelect.scss";
// components
import Map from './map/Map';
import Footer from './footer/Footer';
import Left from './charts/left/Left';
import FunctionBar from './function-bar/FunctionBar';
import Bottom from './charts/bottom/Bottom';
import MapSelectWindow from './map/mapSelect/MapSelectWindow';
// 第三方
import _ from 'lodash';
import { ReloadOutlined, BarChartOutlined, CloseCircleOutlined } from '@ant-design/icons';
import Drawer from '@/components/drawer/Drawer';
import { Button } from 'antd';
// react-redux
import { connect } from 'react-redux';
// 共享数据获取与处理
import {
  fetchData, fetchOceanScoreAll, fetchUsersTopFive, fetchUserTrajNumsByDay,
  setSelectedUsers, setSelectedByMapClick, setSelectedByCharts
} from '@/app/slice/selectSlice';


class PageSelect extends Component {
  constructor(props) {
    super(props);
    this.state = {
      leftWidth: 0, // 左侧栏宽度
      rightWidth: 0, // footer左侧位置
      bottomHeight: 0, //底部内容高度
      bottomWidth: 0, //底部内容宽度
      chartsReload: {}, // 图表重置
      mapClickReload: {}, // 地图bar3D点击重置
      titleVisible: true, // title 初始化 可见
      functionBarItems: [], // 重置functionBar 按钮内容
      allUsers: [], // 所有用户编号
    };
  }

  // 基于selectedByCharts 和 selectByMapClick 数组内的数据，
  // 填充functionBar内容，从而实现控制 图标重置 和 点击重置 的显示与否
  getFunctionBarItems = () => {
    const functionBarItems = _.cloneDeep(this.state.functionBarItems);
    // 获取index
    let chartIndex = -1; let clickIndex = -1;
    functionBarItems.forEach((item) => {
      if (item.text === '图表重置') {
        chartIndex = functionBarItems.indexOf(item)
      }
      if (item.text === '点击重置') {
        clickIndex = functionBarItems.indexOf(item)
      }
    });

    // 根据 charts 数组中数据变化 => 处理 functionBarItems内对应的object
    if (this.props.selectedByCharts.length === 0) { // 如果长度为0
      if (chartIndex !== -1) { // 如果有，则删除
        functionBarItems.splice(chartIndex, 1);
      }
    } else { // 如果长度不为0
      if (chartIndex === -1) { // 如果没有，则添加
        functionBarItems.push({
          text: '图表重置',
          icon: <BarChartOutlined />,
          onClick: () => { this.setState({ chartsReload: {} }) },
        })
      }
    }

    if (this.props.selectedByMapClick.length === 0) {
      if (clickIndex !== -1) {
        functionBarItems.splice(clickIndex, 1);
      }
    } else {
      if (clickIndex === -1) {
        functionBarItems.push({
          text: '点击重置',
          icon: <ReloadOutlined />,
          onClick: () => {
            this.props.setSelectedByMapClick([]);
            this.setState({ mapClickReload: {} })
          }
        })
      }
    }
    this.setState({
      functionBarItems: functionBarItems
    })
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
    return result;
  };

  // 关闭标题
  closeTitle = () => {
    // 后标题消失
    this.setState({
      titleVisible: false
    })
  }

  // 获取所有人员编号，用于初始和清空所有选择时在地图上展示所有用户
  initSelected = (OceanScoreAll) => {
    const allUsers = Object.values(OceanScoreAll).reduce((prev, cur) => {
      const id = parseInt(cur.人员编号);
      if (!prev.includes(id)) {
        prev.push(id);
      }
      return prev;
    }, [])
    return allUsers;
  }

  componentDidMount() {
    // 请求数据(伪)
    this.props.fetchData(`${process.env.PUBLIC_URL}/mock/ocean_score.json`);

    // 请求数据(真)
    // this.props.fetchOceanScoreAll();
    this.props.fetchUsersTopFive();
    this.props.fetchUserTrajNumsByDay();

    //返回各组件的边界位置，用于日历、map等组件的布局
    const leftWidth = document.querySelector('.left').getBoundingClientRect().right;
    const rightWidth = document.querySelector('.footer-bar').getBoundingClientRect().right - document.querySelector('.footer-bar').getBoundingClientRect().left;
    const bottomHeight = document.querySelector('.bottom').getBoundingClientRect().height;
    const bottomWidth = document.querySelector('.bottom').getBoundingClientRect().width;

    this.setState({
      leftWidth: leftWidth,
      rightWidth: rightWidth,
      bottomHeight: bottomHeight,
      bottomWidth: bottomWidth,
    })
    // 初始化 functionBar 中的按钮内容
    this.getFunctionBarItems();
  }

  componentDidUpdate(prevProps, prevState) {
    // 如果大五人格数据获取成功，则前后 OceanScoreAll 不同
    if (!_.isEqual(this.props.OceanScoreAll, prevProps.OceanScoreAll)) {
      // 获取所有人员的编号，初始化 selectedUsers 数组
      const allUsers = this.initSelected(this.props.OceanScoreAll);
      this.props.setSelectedUsers(allUsers);
      this.setState({
        allUsers: allUsers
      })
    }

    if ( // 求出三个列表最终筛选的用户
      !_.isEqual(prevProps.selectedByHistogram, this.props.selectedByHistogram) ||
      !_.isEqual(prevProps.selectedByScatter, this.props.selectedByScatter) ||
      !_.isEqual(prevProps.selectedByParallel, this.props.selectedByParallel)
    ) {
      const result = this.handleIntersection(
        this.props.selectedByHistogram,
        this.props.selectedByScatter,
        this.props.selectedByParallel);
      this.props.setSelectedByCharts(result);
    };

    if ( // 求出最终选择的用户
      !_.isEqual(prevProps.selectedByCharts, this.props.selectedByCharts) ||
      !_.isEqual(prevProps.selectedByCalendar, this.props.selectedByCalendar) ||
      !_.isEqual(prevProps.selectedByMapBrush, this.props.selectedByMapBrush) ||
      !_.isEqual(prevProps.selectedByMapClick, this.props.selectedByMapClick)
    ) {
      const result = this.handleIntersection(
        this.props.selectedByCharts, this.props.selectedByCalendar,
        this.props.selectedByMapBrush, this.props.selectedByMapClick);
      // 如果全都清空了，则设置为所有用户，以便在 map 上展示
      const users = result.length === 0 ? this.state.allUsers : result
      this.props.setSelectedUsers(users);
    };

    // 控制 图标重置 和 点击重置 的显示与否
    if (
      !_.isEqual(prevProps.selectedByCharts, this.props.selectedByCharts) ||
      !_.isEqual(prevProps.selectedByMapClick, this.props.selectedByMapClick)
    ) {
      this.getFunctionBarItems();
    }

    // 是否隐藏标题
    if (!_.isEqual(prevState.titleVisible, this.state.titleVisible)) {
      document.querySelector('.title').style.display = 'none' // 隐藏标题
      this.setState({}) // 重新渲染
    }
  }

  render() {
    return (
      <div className="select-page-ctn">
        <div className="center">
          <span className='title'>
            2019年深圳市私家车用户出行位置Top5地图
            <Button className='button'
              type="primary"
              icon={<CloseCircleOutlined />}
              shape='circle'
              size='small'
              onClick={this.closeTitle}
            />
          </span>
          <Map leftWidth={this.state.leftWidth} bottomHeight={this.state.bottomHeight} rightWidth={this.state.rightWidth} />
          <div className="inner">
            <div className="top-bracket"></div>
            <div className="bottom">
              <Bottom
                bottomHeight={this.state.bottomHeight}
                bottomWidth={this.state.bottomWidth}
              />
            </div>
          </div>
        </div>
        <div className="left">
          <Left
            width={this.state.leftWidth}
            chartsReload={this.state.chartsReload} />
        </div>
        <div className="footer-bar" >
          <Drawer
            render={() => (<Footer setRoutes={this.props.setRoutes} />)}
            type="right"
            width={200}
            initVisible={true}
          />
        </div>
        <FunctionBar functionBarItems={this.state.functionBarItems} left={this.state.leftWidth} />
        <MapSelectWindow right={this.state.rightWidth} bottom={this.state.bottomHeight} />
      </div>
    )
  }
}

const mapStateToProps = (state) => {
  return {
    OceanScoreAll: state.select.OceanScoreAll,
    UsersTopFive: state.select.UsersTopFive,
    UserTrajNumsByDay: state.select.UserTrajNumsByDay,
    selectedByHistogram: state.select.selectedByHistogram,
    selectedByScatter: state.select.selectedByScatter,
    selectedByParallel: state.select.selectedByParallel,
    selectedByCharts: state.select.selectedByCharts,
    selectedByCalendar: state.select.selectedByCalendar,
    selectedBySlider: state.select.selectedBySlider,
    selectedByMapBrush: state.select.selectedByMapBrush,
    selectedByMapClick: state.select.selectedByMapClick,
  }
}

const mapDispatchToProps = (dispatch) => {
  return {
    fetchData: (url) => dispatch(fetchData(url)),
    fetchOceanScoreAll: () => dispatch(fetchOceanScoreAll()),
    fetchUsersTopFive: () => dispatch(fetchUsersTopFive()),
    fetchUserTrajNumsByDay: () => dispatch(fetchUserTrajNumsByDay()),
    setSelectedByCharts: (payload) => dispatch(setSelectedByCharts(payload)),
    setSelectedUsers: (payload) => dispatch(setSelectedUsers(payload)),
    setSelectedByMapClick: (payload) => dispatch(setSelectedByMapClick(payload))
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(PageSelect);