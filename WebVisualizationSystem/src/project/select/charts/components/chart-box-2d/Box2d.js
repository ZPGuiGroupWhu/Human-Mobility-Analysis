import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Store from '@/store';
import './Box2d.scss';
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
    // state
    this.state = {
      isVisible: true, // 是否可视图表
      data: null, // 数据源
      curData: null, // 当前展示的数据
      xAxis: this.defaultXAxis,
      yAxis: this.defaultYAxis,
    };
  }

  getXAxis = (val) => { this.setState({ xAxis: val }) }; // 获取 x 轴类型
  getYAxis = (val) => { this.setState({ yAxis: val }) }; // 获取 y 轴类型

  handleTypeJudge = (data, targetType) => (Object.prototype.toString.call(data) === targetType); // 判断数据类型

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

  // 重置回初始状态
  reState = () => {
    this.setState({
      isVisible: true,
      xAxis: this.defaultXAxis,
      yAxis: this.defaultYAxis,
    })
    this.setCurData(this.state.data, this.defaultXAxis, this.defaultYAxis);
  }

  componentDidMount() {
  }

  componentDidUpdate(prevProps, prevState) {
    // 数据受控
    if (!_.isEqual(this.state.data, this.context.state.allData)) {
      this.setState({
        data: _.cloneDeep(this.context.state.allData),
      })
    }

    // 数据源改变
    if (!_.isEqual(prevState.data, this.state.data)) {
      this.setCurData(this.state.data, this.state.xAxis, this.state.yAxis);
    }
    // 选项改变
    if ((this.state.xAxis !== prevState.xAxis) || (this.state.yAxis !== prevState.yAxis)) {
      this.setCurData(this.state.data, this.state.xAxis, this.state.yAxis);
    }

    if (prevState.isReload !== this.state.isReload) {
      this.reState();
    }
  }

  render() {
    return (
      <div className="chart-box2d-ctn">
        <div className="title-bar">
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
              xAxisName: this.state.xAxis,
              yAxisName: this.state.yAxis,
            })}
          </div>
        </CSSTransition>
      </div>
    );
  }
}

Box2d.contextType = Store;

export default Box2d;