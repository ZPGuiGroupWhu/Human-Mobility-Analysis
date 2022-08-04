import React, { Component } from 'react';
import _ from 'lodash';

export const filterByAxis = (...params) => WrappedComponent => {
  class FilterByAxis extends Component {
    /**
     * props
     * @param {object} data - 数据源
     * @param {string} xAxis - x轴
     * @param {string} yAxis - y轴
     * @param {string} defaultXAxis - 初始x轴
     * @param {string} defaultYAxis - 初始y轴
     */
    constructor(props) {
      super(props);
      this.firstRender = false;
      this.id = props.id;
      this.state = {
        data: null,
      }
    }


    // 依据选择项筛选生成当前视图的渲染数据
    getData = (data, xAxis, yAxis) => {
      try {
        if (!data || !xAxis || !yAxis) return null;
        return Object.values(data).map(obj => {
          return [xAxis, yAxis, '人员编号'].reduce((prev, item) => {
            return [...prev, obj[item] || 0]
          }, []);
        })
      } catch (err) {
        console.log(err);
      }
    }
    // 模拟生成权重 dim = 4
    getDataWeight = (curData) => {
      if (!curData) return null;
      return curData.map(item => ([...item, item[0] + item[1]]));
    }
    // 数据存储
    setData = (data, xAxis, yAxis) => {
      return this.getDataWeight(this.getData(data, xAxis, yAxis))
    }

    handleInit = () => {
      this.setState({
        data: this.setData(this.props.data, this.props.defaultXAxis, this.props.defaultYAxis),
      })
    }


    componentDidUpdate(prevProps, prevState) {
      if (!this.props.data) return;

      // 初次渲染视图
      if (!_.isEqual(this.props.data, prevProps.data) && !this.firstRender) {
        this.handleInit();
        this.firstRender = true;
      }

      // 数据源更新 - 触发重渲染
      if (!_.isEqual(prevProps.data, this.props.data)) {
        this.setState({
          data: this.setData(this.props.data, this.props.xAxis, this.props.yAxis),
        })
      }

      // 坐标轴更新 - 触发重渲染
      if ((this.props.xAxis !== prevProps.xAxis) || (this.props.yAxis !== prevProps.yAxis)) {
        this.setState({
          data: this.setData(this.props.data, this.props.xAxis, this.props.yAxis),
        })
      }

      // 重置坐标轴
      if (this.props.isReload !== prevProps.isReload) {
        this.handleInit()
      }
    }

    render() {
      const { defaultXAxis, defaultYAxis, ...passThroughProps } = this.props;

      return (
        <WrappedComponent {...passThroughProps} {...this.state} />
      );
    }
  }

  return FilterByAxis;
}