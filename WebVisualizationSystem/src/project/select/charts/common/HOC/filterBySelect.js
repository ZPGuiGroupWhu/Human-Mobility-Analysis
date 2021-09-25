import React, { Component } from 'react';
import _ from 'lodash';
import Store from '@/store';

export const filterBySelect = (...params) => WrappedComponent => {
  class FilterBySelect extends Component {
    static contextType = Store;
    static defaultProps = {
      connect: false,
    };

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
        prevSelectedUsers: [], // 用户的历史筛选记录(用于 diff vdom)
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
      const { allData, selectedUsers } = this.context.state;
      this.setState(prev => ({
        prevSelectedUsers: selectedUsers,
        data: _.cloneDeep(this.handleEmptyArray(selectedUsers) ?
          allData :
          this.getDataBySelectedUsers(allData, selectedUsers))
      }));
    }

    componentDidUpdate(prevProps, prevState) {
      // 数据未请求成功
      if (!this.props.reqSuccess) return;

      // 初次渲染
      if (this.props.reqSuccess !== prevProps.reqSuccess) {
        this.updateData();
      }

      // context 中的 selectedUsers 发生变化时，触发图表数据更新
      if (!_.isEqual(prevState.prevSelectedUsers, this.context.state.selectedUsers)) {
        // 数据更新时机：当前筛选图表在进一步筛选时更新，联动图表在刷选结束时更新
        if (this.context.state.curId === this.id) {
          if (this.props.isAxisChange !== prevProps.isAxisChange) {
            this.updateData();
          }
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

  return FilterBySelect;
}