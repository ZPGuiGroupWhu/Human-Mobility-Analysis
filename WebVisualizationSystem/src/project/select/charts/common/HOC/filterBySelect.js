import React, { Component } from 'react';
import _ from 'lodash';
// react-redux
import { connect } from 'react-redux';

export const filterBySelect = (...params) => WrappedComponent => {
  class FilterBySelect extends Component {
    /**
     * props
     * @param {number} id - 当前实例标签
     * @param {object} isBrushEnd - 是否刷选结束
     * @param {object} isAxisChange - 筛选条件是否改变
     */
    constructor(props) {
      super(props);
      this.id = props.id;
      this.state = {
        data: null, // 数据源
        isFirst: false,
      }
    }

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
      try {
        return arr.map(idx => {
          return Object.values(data).find(item => (item['人员编号'] === idx));
        })
      } catch (err) {
        console.log(err);
      }
    }

    // 更新数据
    updateData = () => {
      // 订阅 selectedUsers
      const { data, selectedUsers } = this.props;
      this.setState(prev => ({
        data: _.cloneDeep(this.handleEmptyArray(selectedUsers) ?
          data :
          this.getDataBySelectedUsers(data, selectedUsers))
      }));
    }

    componentDidUpdate(prevProps, prevState) {
      // 数据未请求成功
      if (this.props.reqStatus !== 'succeeded') return;

      // 初次渲染
      if (this.props.reqStatus === 'succeeded' && !this.state.isFirst) {
        this.updateData();
        this.setState({
          isFirst: true,
        })
      }

      // selectedUsers 发生变化时，触发图表数据更新
      if (!_.isEqual(prevProps.selectedUsers, this.props.selectedUsers)) {
        // 数据更新时机：当前筛选图表在一定延迟后更新，联动图表在刷选结束时更新，从而实现同步刷新
        if (this.props.curId === this.id) {
          // if (this.props.isAxisChange === prevProps.isAxisChange) {
            setTimeout(() => {
              this.updateData();
            }, 200) 
          // }
        } else {
          this.updateData();
        }
      }
    }

    render() {
      const { id, isAxisChange, ...passThroughProps } = this.props;

      return (
        <WrappedComponent {...passThroughProps} {...this.state} />
      );
    }
  }

  const mapStateToProps = (state) => {
    return {
      curId: state.select.curId,
      data: state.select.OceanScoreAll, // 图标数据源
      selectedUsers: state.select.selectedUsers,
      reqStatus: state.select.OceanReqStatus,
    }
  }

  return connect(mapStateToProps, null)(FilterBySelect);
}