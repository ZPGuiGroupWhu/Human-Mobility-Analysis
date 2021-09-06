import React, { Component } from 'react';
import axios from 'axios';
import Store from '@/store';

export const reqData = (url) => (WrappedComponent) => {
  class ReqData extends Component {
    constructor(props) {
      super(props);
      this.state = {
        reqSuccess: false,
      }
    }

    // 请求数据
    getData = (url) => {
      axios.get(url).then(
        res => {
          this.context.dispatch({ type: 'setAllData', payload: res.data });
          this.setState({ reqSuccess: true }); // 数据请求成功
        }
      )
    }

    componentDidMount() {
      this.getData(url); // 请求数据
    }

    render() {
      return (
        <WrappedComponent {...this.state} {...this.props} />
      )
    }
  }

  ReqData.contextType = Store;

  return ReqData;
}