import React, { Component } from 'react';
import './Box.scss';
import '@/project/border-style.scss';
import {
  CompressOutlined,
  ExpandOutlined,
} from '@ant-design/icons';
import { Space } from 'antd';
import _ from 'lodash';
import Hover from '../../common/Hover';
import DropMenu from '../../common/DropMenu';
import Chart from './Chart';
// react-redux
import { connect } from 'react-redux';
import { 
  setCurId, 
  setSelectedByCharts, 
  setSelectedByHistogram, 
  setSelectedByParallel, 
  setSelectedByScatter  } from '@/app/slice/selectSlice';

class Box extends Component {
  // icon 通用配置
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }

  /**
   * props
   * @param {boolean} reqSuccess - 源数据是否请求成功
   * @param {object} isReload - 是否重置
   * @param {array || string} xAxis - x轴选项列表(字段)
   * @param {array || string} yAxis - y轴选项列表(字段)
   * @param {number} id - 实例标识
   * @param {isBrushEnd} isBrushEnd - 是否刷选结束
   * @param {function} handleBrushEnd - 刷选结束事件
   */
  constructor(props) {
    super(props);
    this.defaultXAxis = this.handleTypeJudge(props.xAxis, '[object Array]') ? props.xAxis[0] : props.xAxis; // 初始默认 xAxis
    this.defaultYAxis = this.handleTypeJudge(props.yAxis, '[object Array]') ? props.yAxis[1] : props.yAxis; // 初始默认 yAxis
    // state
    this.state = {
      isVisible: true, // 是否可视图表
      xAxis: this.defaultXAxis, // x 轴
      yAxis: this.defaultYAxis, // y 轴
      isAxisChange: {}, // 筛选条件是否改变
    };
  }

  // 鼠标移入/移除组件 - 标识当前操作图表的 id
  onMouseEnter = () => {
    this.props.setCurId(this.props.id);
  }
  onMouseLeave = () => {
    this.props.setCurId(-1);
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }

  getXAxis = (val) => { this.setState({ xAxis: val }) }; // 获取 x 轴类型
  getYAxis = (val) => { this.setState({ yAxis: val }) }; // 获取 y 轴类型

  // 判断数据类型
  handleTypeJudge = (data, targetType) => (Object.prototype.toString.call(data) === targetType);
  // 判断数组是否为空
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
  };


  // 重置
  handleReload = () => {
    this.setState({
      isVisible: true,
      xAxis: this.defaultXAxis,
      yAxis: this.defaultYAxis,
    });
    if (!this.handleEmptyArray(this.props.selectedByCharts)) {
      this.props.setSelectedByHistogram([]);
      this.props.setSelectedByScatter([]);
      this.props.setSelectedByParallel([]);
    }
  }




  componentDidUpdate(prevProps, prevState) {
    if ((this.state.xAxis !== prevState.xAxis) || (this.state.yAxis !== prevState.yAxis)) {
      this.setState({
        isAxisChange: {},
      })
    }

    // 重置回初始状态
    if (prevProps.isReload !== this.props.isReload) {
      this.handleReload();
    }
  }

  render() {
    return (
      <div
        className="chart-box-ctn tech-border"
        onMouseEnter={this.onMouseEnter}
        onMouseLeave={this.onMouseLeave}
      >
        <div className="title-bar">
          {
            this.state.isVisible ?
              <>
                <div className="axis-title">
                  <span style={{ fontWeight: 'bold' }}>X:</span>
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
                </div>
                <div className="axis-title">
                  <span style={{ fontWeight: 'bold' }}>Y:</span>
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
                </div>
              </> :
              <span style={{ marginLeft: '5px', fontWeight: 'bold' }}>{`${this.state.xAxis}-${this.state.yAxis}`}</span>
          }


          <div className="func-btns">
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
        {/* 图表 */}
        <Chart
          isVisible={this.state.isVisible} // 控制 Chart 可视
          id={this.props.id} // 实例id
          handleBrushEnd={this.props.handleBrushEnd} // 刷选结束事件
          isBrushEnd={this.props.isBrushEnd} // 刷选结束
          isAxisChange={this.state.isAxisChange} // 坐标轴改变
          xAxis={this.state.xAxis} // x轴
          yAxis={this.state.yAxis} // y轴
          defaultXAxis={this.defaultXAxis} // 默认初始 x 轴
          defaultYAxis={this.defaultYAxis} // 默认初始 y 轴
          isReload={this.props.isReload} // 重置
          render={this.props.render} // render
        />
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    selectedByCharts: state.select.selectedByCharts,
  }
}

const mapDispatchToProps = (dispatch) => {
  return {
    setCurId: (payload) => dispatch(setCurId(payload)),
    setSelectedByHistogram: (payload) => dispatch(setSelectedByHistogram(payload)),
    setSelectedByScatter: (payload) => dispatch(setSelectedByScatter(payload)),
    setSelectedByParallel: (payload) => dispatch(setSelectedByParallel(payload)),
    setSelectedByCharts: (payload) => dispatch(setSelectedByCharts(payload))
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(Box);