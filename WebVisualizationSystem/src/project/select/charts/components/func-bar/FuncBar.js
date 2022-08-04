import React, { Component } from 'react';
import './FuncBar.scss';
import '@/project/border-style.scss';
import { Space } from 'antd';
import _ from 'lodash';

import IconButton from '../../common/IconButton';
import { reloadWhite, reloadActive } from '@/assets/select-charts/reload';




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
    this.props.handleConnect();
  }

  handleReload = () => {
    this.props.handleReload();
  }

  render() {
    return (
      <div className="chart-funcbar-ctn tech-border">
        <div className="title-bar">
          <div className="func-btns">
            <Space>
              <IconButton
                text="重置"
                actImage={reloadActive}
                noActImage={reloadWhite}
                onClick={this.handleReload}
                imgHeight={'20px'}
                isReserveActive={false}
              />
            </Space>
          </div>
        </div>
      </div>
    );
  }
}

export default FuncBar;