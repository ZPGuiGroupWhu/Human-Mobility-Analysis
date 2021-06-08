import React, { useState, useEffect, useMemo } from 'react';
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
  // const formatData = formatter(data);
  const formatData = useMemo(() => {return formatter(data)}, [data]);

  const [targetKeys, setTargetKeys] = useState([]);
  const [selectedKeys, setSelectedKeys] = useState([]);

  function handleChange(nextTargetKeys, direction, moveKeys) {
    setTargetKeys(nextTargetKeys);
  }

  function handleSelectChange(sourceSelectedKeys, targetSelectedKeys) {
    setSelectedKeys([...sourceSelectedKeys, ...targetSelectedKeys]);
  }

  useEffect(() => {
    // 每次 formatData 都是新生成的数组，内存地址不同，因此依赖项若设置为 formatData 会死循环, 此处设置为 data。
    // 或者将 formatData 用 useMemo 设为不可变对象。
    setTargetKeys(() => (formatData.map(item => item.key)));
  }, [formatData])

  useEffect(() => {
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
