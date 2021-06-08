import React, { useState, useEffect } from 'react';
import { Transfer } from 'antd';


/*
data format
export interface TransferItem {
  key: string;
  title: string;
  description?: string;
  disabled?: boolean;
}
*/
function formatter(data) {
  const res = data
    .sort((a, b) => (a.distance - b.distance))
    .map((item, idx) => {
      return {
        key: item.id.toString(),
        title: `${item.distance} km`,
      }
    })
  return res;
}

export default function TransferSelector(props) {
  const { data, setData } = props;
  const formatData = formatter(data);
  const orgTargetKeys = formatData.map(item => item.key);

  const [targetKeys, setTargetKeys] = useState(orgTargetKeys);
  const [selectedKeys, setSelectedKeys] = useState([]);

  function handleChange(nextTargetKeys, direction, moveKeys) {
    setTargetKeys(nextTargetKeys);
  }

  function handleSelectChange(sourceSelectedKeys, targetSelectedKeys) {
    setSelectedKeys([...sourceSelectedKeys, ...targetSelectedKeys]);
  }

  useEffect(() => {
    console.log(data.filter(item => (targetKeys.includes(item.id.toString()))));
    setData(data.filter(item => (targetKeys.includes(item.id.toString()))))
  }, [targetKeys])

  return (
    <Transfer
      dataSource={formatData}
      // 左标题，右标题
      titles={['未选中项', '选中项']}
      // 显示在右侧框数据的 key 集合 {string[]}
      targetKeys={targetKeys}
      // 当前选中项(勾选项)
      selectedKeys={selectedKeys}
      // 选项在两栏之间转移时的回调函数
      onChange={handleChange}
      onSelectChange={handleSelectChange}
      render={item => item.title}
      // 分页
      pagination
      style={{ width: '100%' }}
    />
  )
}
