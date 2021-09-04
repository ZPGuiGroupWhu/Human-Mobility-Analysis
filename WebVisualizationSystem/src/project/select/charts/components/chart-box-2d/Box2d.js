import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Store from '@/store';
import './Box2d.scss';
import {
  CompressOutlined,
  ExpandOutlined,
  ReloadOutlined,
  SnippetsOutlined,
} from '@ant-design/icons';
import { Space } from 'antd';
import { CSSTransition } from 'react-transition-group';
import _ from 'lodash';
import Hover from '../../common/Hover';
import DropMenu from '../../common/DropMenu';


class Box2d extends Component {
  // icon 通用配置
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }

  constructor(props) {
    super(props);
    this.defaultXAxis = this.handleTypeJudge(props.xAxis, '[object Array]') ? props.xAxis[0] : props.xAxis; // 初始默认 xAxis
    this.defaultYAxis = this.handleTypeJudge(props.yAxis, '[object Array]') ? props.yAxis[1] : props.yAxis; // 初始默认 yAxis
    this.defaultWithFilter = props.withFilter;
    this.prevSelectedUsers = null; // 历史 context 记录 - 监听 context 内容变化
    this.prevData = null;
    // state
    this.state = {
      isVisible: true, // 是否可视图表
      data: null, // 数据源
      curData: null, // 当前展示的数据
      xAxis: this.defaultXAxis,
      yAxis: this.defaultYAxis,
      withFilter: this.defaultWithFilter,
      prevFilter: this.defaultWithFilter,
    };
  }

  getXAxis = (val) => { this.setState({ xAxis: val }) }; // 获取 x 轴类型
  getYAxis = (val) => { this.setState({ yAxis: val }) }; // 获取 y 轴类型

  handleTypeJudge = (data, targetType) => (Object.prototype.toString.call(data) === targetType); // 判断数据类型
  handleEmptyArray = (arr) => {
    try {
      if (!Array.isArray(arr)) throw new Error('input should be Array Type');
      if (arr.length === 0) {
        return true;
      } else {
        arr.reduce((prev, item) => {
          if (!Array.isArray(item)) return false;
          return prev && this.handleEmptyArray(item);
        }, true)
      }
    } catch (err) {
      console.log(err);
    }
  }; // 判断数组是否为空


  // 根据人员编号筛选数据
  getDataBySelectedUsers = (data, arr) => {
    return arr.map(idx => {
      return Object.values(data).find(item => (item['人员编号'] === idx));
    })
  }

  // 依据选择项筛选生成当前视图的渲染数据
  getCurData = (data, xAxis, yAxis) => {
    if (!data || !xAxis || !yAxis) return null;
    return Object.values(data).map(obj => {
      return [xAxis, yAxis, '人员编号'].reduce((prev, item) => {
        return [...prev, obj[item]]
      }, []);
    })
  }
  // 模拟生成权重 dim = 4
  getCurDataWeight = (curData) => {
    if (!curData) return null;
    return curData.map(item => ([...item, item[0] + item[1]]));
  }
  // 数据存储
  setCurData = (data, xAxis, yAxis) => {
    this.setState({
      curData: this.getCurDataWeight(this.getCurData(data, xAxis, yAxis)),
    })
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }

  forbiddenFilter = () => {
    this.setState(prev => {
      return {
        withFilter: false
      }
    });
  }

  reopenFilter = () => {
    this.setState({
      withFilter: this.state.prevFilter,
    })
  }

  // 重置回初始状态
  reState = () => {
    this.setState({
      isVisible: true,
      withFilter: this.defaultWithFilter,
      xAxis: this.defaultXAxis,
      yAxis: this.defaultYAxis,
    })
    this.handleInit();
    this.context.dispatch({
      type: 'setSelectedUsers',
      payload: []
    })
  }

  handleInit = () => {
    const data = _.cloneDeep(this.context.state.allData);
    this.setState({
      data,
    })
    this.setCurData(data, this.defaultXAxis, this.defaultYAxis);
    this.prevData = data;
  }


  componentDidUpdate(prevProps, prevState) {
    if (!this.props.reqSuccess) return; // 数据未请求成功
    if (this.props.reqSuccess !== prevProps.reqSuccess) {
      this.handleInit();
    } // 初次渲染视图

    if (!_.isEqual(this.prevSelectedUsers, this.context.state.selectedUsers)) {
      const { allData, selectedUsers } = this.context.state;
      this.setState(prev => {
        return {
          data: _.cloneDeep(
            this.handleEmptyArray(selectedUsers) ?
              allData :
              this.getDataBySelectedUsers(prev.withFilter ? this.prevData : allData, selectedUsers)
          ),
        }
      })
      this.prevSelectedUsers = [...selectedUsers];
    }

    // 数据源改变
    // 数据源改变
    if ((this.state.xAxis !== prevState.xAxis) || (this.state.yAxis !== prevState.yAxis)) {
      this.prevData = this.state.data;
      if (this.state.withFilter) {
        this.setCurData(this.state.data, this.state.xAxis, this.state.yAxis);
      } else {
        this.setCurData(this.context.state.allData, this.state.xAxis, this.state.yAxis);
      }
    }

    if (!_.isEqual(this.state.data, prevState.data)) {
      if (this.state.withFilter) {
        this.setCurData(this.state.data, this.state.xAxis, this.state.yAxis);
      }
    }

    if (prevState.isReload !== this.state.isReload) {
      this.reState();
    }
  }

  render() {
    return (
      <div className="chart-box2d-ctn">
        <div className="title-bar">
          <span>X:</span>
          {
            this.handleTypeJudge(this.props.xAxis, '[object Array]') ?
              <DropMenu
                defaultValue={this.state.xAxis}
                value={this.state.xAxis}
                items={this.props.xAxis}
                getSelectItem={this.getXAxis}
              /> :
              <span className="text">{this.state.xAxis}</span>
          }
          <span>Y:</span>
          {
            this.handleTypeJudge(this.props.yAxis, '[object Array]') ?
              <DropMenu
                defaultValue={this.state.yAxis}
                value={this.state.yAxis}
                items={this.props.yAxis}
                getSelectItem={this.getYAxis}
              /> :
              <span className="text">{this.state.yAxis}</span>
          }
          <div className="func-btns">
            <Space>
              <Hover isReload={this.state.isReload} isClicked={this.defaultWithFilter}>
                {
                  ({ isHovering, isClicked }) => (
                    <SnippetsOutlined
                      style={{
                        ...this.iconStyle,
                        display: this.props.filterable ? '' : 'none',
                        color: (isHovering || this.state.withFilter) ? '#05f8d6' : '#fff'
                      }}
                      onClick={() => {
                        this.setState(prev => {
                          return {
                            withFilter: !prev.withFilter,
                            prevFilter: !prev.prevFilter,
                          }
                        })
                      }}
                    />
                  )
                }
              </Hover>
              <Hover>
                {
                  ({ isHovering }) => (
                    <ReloadOutlined
                      style={{
                        ...this.iconStyle,
                        color: isHovering ? '#05f8d6' : '#fff'
                      }}
                      onClick={() => { this.setState({ isReload: {} }) }}
                    />
                  )
                }
              </Hover>
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
        <CSSTransition
          in={this.state.isVisible}
          timeout={300}
          classNames='chart'
          onEnter={(node) => { node.style.setProperty('display', '') }}
          onExiting={(node) => { node.style.setProperty('display', 'none') }}
        >
          <div
            className="chart-content"
          >
            {this.props.children(this.state.curData, {
              withFilter: this.state.withFilter,
              xAxisName: this.state.xAxis,
              yAxisName: this.state.yAxis,
              forbiddenFilter: this.forbiddenFilter,
              reopenFilter: this.reopenFilter,
            })}
          </div>
        </CSSTransition>
      </div>
    );
  }
}

Box2d.contextType = Store;

Box2d.propTypes = {
  reqSuccess: PropTypes.bool.isRequired,
}

Box2d.defaultProps = {
  filterable: false,
  withFilter: true,
}

export default Box2d;