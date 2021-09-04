import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './FuncBar.scss';
import { Space } from 'antd';
import _ from 'lodash';

import IconButton from '../../common/IconButton';
import { connectWhite, connectActive } from '@/assets/select-charts/connect';




class FuncBar extends Component {
  // icon 通用配置
  iconStyle = {
    fontSize: '13px',
    color: '#fff',
  }

  constructor(props) {
    super(props);
    // state
    this.state = {
      isVisible: true, // 是否可视图表
    };
  }

  // 内容展开/关闭
  setChartVisible = () => {
    this.setState(prev => ({
      isVisible: !prev.isVisible
    }))
  }

  handleConnect = () => {
    console.log('connect');
  }

  render() {
    return (
      <div className="chart-box1d-ctn">
        <div className="title-bar">
          <div className="func-btns">
            <Space>
              <IconButton actImage={connectActive} noActImage={connectWhite} onClick={ this.handleConnect } />
            </Space>
          </div>
        </div>
      </div>
    );
  }
}

FuncBar.propTypes = {
  reqSuccess: PropTypes.bool.isRequired,
}

FuncBar.defaultProps = {
  filterable: false,
  withFilter: true,
}

export default FuncBar;