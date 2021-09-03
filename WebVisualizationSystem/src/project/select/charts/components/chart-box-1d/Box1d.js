import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Store from '@/store';
import './Box1d.scss';
import {
  CompressOutlined,
  ExpandOutlined,
  ReloadOutlined,
  SnippetsOutlined,
} from '@ant-design/icons';
import { Space, Select } from 'antd';
import { CSSTransition } from 'react-transition-group';
import _ from 'lodash';
import Hover from '../../common/Hover';

const { Option } = Select;
class DropMenu extends Component {
  getSelectItem = (value) => {
    this.props.getSelectItem(value)
  }

  render() {
    return (
      <Select
        defaultValue={this.props.defaultValue}
        value={this.props.value}
        style={{ width: 100 }}
        bordered={false} // 是否显示边框
        showArrow={false} // 是否显示箭头
        showSearch={false} // 是否启用搜索
        onSelect={(value) => { this.getSelectItem(value) }} // 触发选中时的回调函数
      >
        {
          this.props.items.map((item, idx) => (
            <Option value={item} key={idx}>{item}</Option>
          ))
        }
      </Select>
    )
  }
}


class Box1d extends Component {
  // icon 通用配置
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }

  constructor(props) {
    super(props);
    this.defaultAxis = this.handleTypeJudge(props.axis, '[object Array]') ? props.axis[0] : props.axis; // 初始默认 axis
    this.prevSelectedUsers = null; // 历史 context 记录 - 监听 context 内容变化
    this.prevData = null;
    // state
    this.state = {
      isVisible: true, // 是否可视图表
      data: null, // 数据源
      curData: null, // 当前展示的数据
      axis: this.defaultAxis,
      withFilter: false, // 是否启用过滤
    };
  }

  getAxis = (val) => { this.setState({ axis: val }) };

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
  getCurData = (data, axis) => {
    if (!data || !axis) return null;
    return Object.values(data).map(obj => {
      return [axis, '人员编号'].reduce((prev, item) => {
        return [...prev, obj[item]]
      }, []);
    })
  }
  // 数据存储
  setCurData = (data, axis) => {
    this.setState({
      curData: this.getCurData(data, axis),
    })
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }

  // 重置回初始状态
  reState = () => {
    this.setState({
      isVisible: true,
      withFilter: false,
      axis: this.defaultAxis,
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
    this.setCurData(data, this.defaultAxis);
    this.prevData = data;
  }

  componentDidUpdate(prevProps, prevState) {
    if (!this.props.reqSuccess) return; // 数据未请求成功
    if (this.props.reqSuccess !== prevProps.reqSuccess) {
      this.handleInit();
    } // 初次渲染视图

    // 筛选
    if (!_.isEqual(this.prevSelectedUsers, this.context.state.selectedUsers)) {
      const { allData, selectedUsers } = this.context.state; // 订阅 selectedUsers
      this.setState(prev => {
        return {
          data: _.cloneDeep(
            this.handleEmptyArray(selectedUsers) ?
              allData :
              this.getDataBySelectedUsers(prev.withFilter ? this.prevData : allData, selectedUsers))
        }
      });
      this.prevSelectedUsers = [...selectedUsers];
    }
    // 数据源改变
    if (this.state.axis !== prevState.axis) {
      this.prevData = this.state.data;
      if (this.state.withFilter) {
        this.setCurData(this.state.data, this.state.axis);
      } else {
        this.setCurData(this.context.state.allData, this.state.axis);
      }
    }

    if (prevState.isReload !== this.state.isReload) {
      this.reState();
    }
  }

  render() {
    return (
      <div className="chart-box1d-ctn">
        <div className="title-bar">
          {
            this.handleTypeJudge(this.props.axis, '[object Array]') ?
              <DropMenu
                defaultValue={this.state.axis}
                value={this.state.axis}
                items={this.props.axis}
                getSelectItem={this.getAxis}
              /> :
              <span className="text">{this.state.axis}</span>
          }
          <div className="func-btns">
            <Space>
              <Hover isReload={this.state.isReload}>
                {
                  ({ isHovering, isClicked }) => (
                    <SnippetsOutlined
                      style={{
                        ...this.iconStyle,
                        display: this.props.filterable ? '' : 'none',
                        color: (isHovering || isClicked) ? '#05f8d6' : '#fff'
                      }}
                      onClick={() => { this.setState(prev => ({ withFilter: !prev.withFilter })) }}
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
              axisName: this.state.axis,
              withFilter: this.state.withFilter,
            })}
          </div>
        </CSSTransition>
      </div>
    );
  }
}

Box1d.contextType = Store;

Box1d.propTypes = {
  reqSuccess: PropTypes.bool.isRequired,
}

Box1d.defaultProps = {
  filterable: false,
}

export default Box1d;