import React, { Component } from 'react';
import { Radio, Space } from 'antd';
import './BtmDrawer.scss';
import _ from 'lodash';
import ShoppingCart from '@/project/analysis/components/shopping/ShoppingCart'; // 购物车

class BtmDrawer extends Component {
  constructor(props) {
    super(props);
    this.radioContents = [
      { id: 1, text: '购物车' },
    ];
    this.state = {
      value: 1,
    }
  }

  onChange = (e) => {
    this.setState({
      value: e.target.value,
    })
  }

  render() {
    return (
      <>
        <div className='btmdrw-radio btmdrw-moudle'>
          <Radio.Group onChange={this.onChange} value={this.state.value}>
            <Space direction="vertical">
              {this.radioContents.map(item => (<Radio value={item.id}>{item.text}</Radio>))}
            </Space>
          </Radio.Group>
        </div>
        <ShoppingCart isSelected={this.state.value === 1} />
      </>
    );
  }
}

export default BtmDrawer;