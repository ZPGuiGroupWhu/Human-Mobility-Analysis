import React from 'react';
import { Select, Empty } from 'antd';

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
      notFoundContent={<Empty image={Empty.PRESENTED_IMAGE_SIMPLE} />}
    >
      {
        data.sort((a, b) => a.distance - b.distance).map(item => {
          return item ? (
            <Option
              key={item.id}
              title={item.distance}
              value={item.id}
            >{`${item.distance.toString()} km`}</Option>
          ) : null;
        })
      }
      <Option></Option>
    </Select>
  )
}
