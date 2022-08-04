import React, { Component } from 'react';
import { Select } from 'antd';

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

export default DropMenu;