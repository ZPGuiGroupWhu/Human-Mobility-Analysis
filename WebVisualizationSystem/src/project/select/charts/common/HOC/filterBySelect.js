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
     * @param {boolean?} connect - 图表间是否联动
     * @param {object} isBrushEnd - 是否刷选结束
     * @param {object} isAxisChange - 筛选条件是否改变
     */
    constructor(props) {
      super(props);
      this.id = props.id;
      this.state = {
        data: null, // 数据源
        privateSelectedUsers: [], // 私有筛选成员编号数组
        historyData: null, // 记录最近一次的数据

        prevSelectedByCalendar: [], // 历史日历筛选结果(限制更新)
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

    updateData = (isConnect) => {
      const { allData, selectedUsers } = this.context.state; // 订阅 selectedUsers
      this.setState(prev => ({
        data: _.cloneDeep(
          this.handleEmptyArray(isConnect ? selectedUsers : this.state.privateSelectedUsers) ?
            allData :
            this.getDataBySelectedUsers(this.state.historyData, isConnect ? selectedUsers : this.state.privateSelectedUsers))
      }));
    }

    componentDidUpdate(prevProps, prevState) {
      if (!this.props.reqSuccess) return; // 数据未请求成功
      if (this.props.reqSuccess !== prevProps.reqSuccess) {
        this.updateData(this.props.isConnect);
        this.setState({ historyData: this.context.state.allData, });
      } // 初次渲染

      // 其余组件筛选时，触发box更新
      if (!_.isEqual(prevState.prevSelectedByCalendar, this.context.state.selectedByCalendar)) {
        this.setState({
          prevSelectedByCalendar: this.context.state.selectedByCalendar
        });
        this.updateData(this.props.connect);
      }

      // 若图表间联动，则调用全局的筛选人员ID，反之调用自身维护的人员ID
      if (this.props.connect) {
        // 数据更新时机：当前筛选图表在进一步筛选时更新，联动图表在刷选结束时更新
        if (this.context.state.curId === this.id) {
          if (this.props.isAxisChange !== prevProps.isAxisChange) {
            this.setState({ historyData: this.state.data, });
            this.updateData(this.props.connect);
          }
        } else {
          if (prevProps.isBrushEnd !== this.props.isBrushEnd) {
            this.updateData(this.props.connect);
          }
        }
      } else {
        if (this.props.isAxisChange !== prevProps.isAxisChange) {
          this.setState({ historyData: this.state.data, });
          this.updateData(this.props.connect);
        }
      }

      // 重置数据
      if (this.props.isReload !== prevProps.isReload) {
        this.setState({ historyData: this.context.state.allData, });
        this.context.dispatch({ type: 'setSelectedUsers', payload: [] });
        this.setState({ data: this.context.state.allData });
      }
    }

    render() {
      const { id, connect, isBrushEnd, isAxisChange, ...passThroughProps } = this.props;

      return (
        <WrappedComponent {...passThroughProps} {...this.state} />
      );
    }
  }

  return FilterBySelect;
}