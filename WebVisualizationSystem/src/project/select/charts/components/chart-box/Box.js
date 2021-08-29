import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './Box.scss';
import { CompressOutlined, ExpandOutlined, RiseOutlined, SnippetsOutlined } from '@ant-design/icons';
import { Space, Select } from 'antd';
import { CSSTransition } from 'react-transition-group';
import _ from 'lodash';

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
      chartVisible: true, // 是否可视图表
      data: null, // 数据源
      sortedData: null, // 排序后的数据源
      curSelectItem: this.defaultSelectItem, // 当前选择项(多源数据时生效)
      isSorted: {}, // 是否开启
      withFilter: false, // 是否开启过滤功能
    };
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      chartVisible: !prev.chartVisible
    }))
  }

  // 选择列表
  getSelectItem = (val) => {
    this.setState({
      curSelectItem: val,
    })
  }

  // 返回选择结果
  findItem = (arr, target) => {
    return arr.find((item) => {
      return item.title === target
    })
  }

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

  componentDidUpdate(prevProps, prevState) {
    if (!_.isEqual(prevProps.data, this.props.data) ||
      !_.isEqual(prevState.curSelectItem, this.state.curSelectItem)) {
      let res = this.findItem(this.props.data, this.state.curSelectItem);
      this.setState({
        data: typeof res.data === 'function' ? res.data() : res.data,
      })
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
              <RiseOutlined
                style={{ ...this.iconStyle, display: this.props.sortable ? '' : 'none' }}
                onClick={() => { this.setState({isSorted: {}}) }}
              />
              <SnippetsOutlined
                style={{ ...this.iconStyle, display: this.props.filterable ? '' : 'none' }}
                onClick={() => { this.setState(prev => ({ withFilter: !prev.withFilter })) }}
              />
              {
                this.state.chartVisible ?
                  <CompressOutlined style={this.iconStyle} onClick={this.setChartVisible} /> :
                  <ExpandOutlined style={this.iconStyle} onClick={this.setChartVisible} />
              }
            </Space>
          </div>
        </div>
        <CSSTransition
          in={this.state.chartVisible}
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