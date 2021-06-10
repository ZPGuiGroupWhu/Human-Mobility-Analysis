import React from 'react';
import { Select } from 'antd';

/**
 * @param {{id: number, distance: number, coords: number[][]}[]} data - 轨迹数组
 * @param {(val: number): any} onSelect - 选中触发的回调函数
 */
export default function SingleTrajSelector(props) {
  const { Option } = Select;
  const { data = [], onSelect = null, onClear = null } = props;

  return (
    <Select
      style={{ width: '100%' }}
      // 允许清除
      allowClear
      onSelect={(val) => onSelect && onSelect(val)}
      onClear={onClear}
      // Option 排序规则
      filterSort={(op1, op2) => (op1.distance - op2.distance)}
    >
      {
        data.map((item, idx) => {
          return item ? (
            <Option
              title={item.distance.toString()}
              value={item.id}
            >{`${item.distance.toString()} km`}</Option>
          ) : null;
        })
      }
      <Option></Option>
    </Select>
  )
}
