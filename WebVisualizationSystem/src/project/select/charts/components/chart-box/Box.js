import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './Box.scss';
import {
  CompressOutlined,
  ExpandOutlined,
  RiseOutlined,
  SnippetsOutlined,
  ReloadOutlined
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
        defaultValue={this.props.items[0]}
        style={{ width: 120 }}
        bordered={false} // 是否显示边框
        showArrow={false} // 是否显示箭头
        showSearch // 是否启用搜索
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


// Box 宽度继承父元素，高度由子元素撑起
class Box extends Component {
  // icon 通用配置
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }

  constructor(props) {
    super(props);
    this.defaultSelectItem = props.dataKey[0];
    // state
    this.state = {
      isVisible: true, // 是否可视图表
      data: null, // 数据源
      sortedData: null, // 存储排序后的数据源
      curSelectItem: this.defaultSelectItem, // 当前选择项(多源数据时生效)
      isSorted: {}, // 是否进行一次排序：对象间比较必为false，确保每次都触发
      isReload: {}, // 是否进行一次重置
      withFilter: false, // 是否开启过滤功能
    };
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }

  // 选择列表
  getSelectItem = (val) => {
    this.setState({
      curSelectItem: val,
    })
  }

  // 返回选择结果
  findItem = (arr, target) => (arr.find((item) => (item.title === target)));

  // 数据排序(按照 dim 维度)
  setSortableData = (data, dim) => {
    try {
      if (!Array.isArray(data)) throw new Error('data should be Array Type');
      if (!dim || (dim >= data.length)) throw new Error('dim Error');
      data.sort((a, b) => (a[dim] - b[dim]));
      this.setState({
        sortedData: _.cloneDeep(data), // 深拷贝，确保数据更新能被监听
      })
    } catch (e) {
      console.log(e);
    }
  }

  getBoxData = (data, curSelectItem) => {
    let res = this.findItem(data, curSelectItem);
    this.setState({
      data: typeof res.data === 'function' ? res.data() : res.data,
    })
  }

  // 重置回初始状态
  reState = () => {
    this.setState({
      isVisible: true,
      withFilter: false,
    })
    this.getBoxData(this.props.data, this.state.curSelectItem);
  }

  componentDidUpdate(prevProps, prevState) {
    // 数据源发生改变 或 筛选项发生改变 - 更换当前 box 的数据源
    if (!_.isEqual(prevProps.data, this.props.data) || !_.isEqual(prevState.curSelectItem, this.state.curSelectItem)) {
      let res = this.findItem(this.props.data, this.state.curSelectItem);
      this.setState({
        data: typeof res.data === 'function' ? res.data() : res.data,
      })
    }

    if (prevState.isReload !== this.state.isReload) {
      this.reState();
    }
  }

  render() {
    return (
      <div className="chart-box-ctn">
        <div className="title-bar">
          {
            this.props.dataKey.length !== 1 ?
              <DropMenu items={this.props.dataKey} getSelectItem={this.getSelectItem} /> :
              <span className="text">{this.props.data[0].title}</span>
          }
          <div className="func-btns">
            <Space>
              <Hover>
                {
                  ({ isHovering }) => (
                    <RiseOutlined
                      style={{
                        ...this.iconStyle,
                        display: this.props.sortable ? '' : 'none',
                        color: isHovering ? '#05f8d6' : '#fff'
                      }}
                      onClick={() => { this.setState({ isSorted: {} }) }} // 触发一次排序
                    />
                  )
                }
              </Hover>
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
                        display: this.props.filterable ? '' : 'none',
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
            {this.props.children(
              this.state.data,
              {
                withFilter: this.state.withFilter,
                isSorted: this.state.isSorted,
                sortedData: this.state.sortedData,
                setSortableData: this.setSortableData,
              }
            )}
          </div>
        </CSSTransition>
      </div>
    );
  }
}

Box.propTypes = {
  sortable: PropTypes.bool,
  filterable: PropTypes.bool,
}

Box.defaultProps = {
  sortable: false,
  filterable: false,
}

export default Box;