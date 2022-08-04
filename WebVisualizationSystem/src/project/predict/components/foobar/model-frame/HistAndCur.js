import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { Button, InputNumber, Space, Select, Tooltip } from 'antd';
import { EyeOutlined, EyeInvisibleOutlined } from '@ant-design/icons';
import './HistAndCur.scss';
import eventBus, { HISTACTION } from '@/app/eventBus';
import { getUserHistoryTraj } from '@/network';

export default function HistAndCur(props) {
  const curShowTrajId = useSelector(state => state.analysis.curShowTrajId); // 当前展示的轨迹 id

  // 是否展示历史轨迹: false-当前未展示轨迹
  const [isShow, setShow] = useState(false);
  const showOnOff = e => { setShow(prev => !prev) }
  const showOn = e => { setShow(true) }


  const [num, setNum] = useState(1); // input框内数值
  const [dayNum, setDayNum] = useState(num); // 需要展示的轨迹天数
  const onInputChange = val => { setNum(val || 0) }

  // 数据请求
  const [data, setData] = useState([]);
  useEffect(() => {
    async function fetchData(id, days) {
      try {
        let data = await getUserHistoryTraj(id, days);
        setData(data);
      } catch (err) {
        console.log(err);
      }
    }
    isShow && fetchData(curShowTrajId, dayNum);
  }, [isShow, curShowTrajId, dayNum])

  // 数据分发
  useEffect(() => {
    if (!data.length) return () => { };
    eventBus.emit(HISTACTION, isShow ? data : [])
  }, [isShow, data])

  // 组件销毁前，取消当前组件操作所有产生的结果
  useEffect(() => {
    return () => {
      setShow(false);
      eventBus.emit(HISTACTION, [])
    }
  }, [])

  // 单位
  const [type, setType] = useState('day');
  const handleSelect = (value) => { setType(value) };

  // 自动根据单位转换输入框内容
  useEffect(() => {
    if (type === 'day') { setNum(Math.ceil(dayNum / 1)); }
    if (type === 'week') { setNum(Math.ceil(dayNum / 7)); }
  }, [type]);

  // 更新展示的轨迹天数
  useEffect(() => {
    if (type === 'day') { setDayNum(num); }
    if (type === 'week') { setDayNum(num * 7); }
  }, [type, num]);

  return (
    <div className="hist-cur-row">
      <Space>
        <InputNumber
          min={0}
          max={28}
          keyboard={true}
          controls={true}
          value={num}
          onChange={val => onInputChange(val)}
          onPressEnter={e => { showOn() }}
          style={{ width: '90%' }}
        />
        <Select
          defaultValue='day'
          bordered={false} showArrow={false} style={{ color: '#fff', width: '40px' }}
          onSelect={handleSelect}
        >
          <Select.Option value='day'>天</Select.Option>
          <Select.Option value='week'>周</Select.Option>
        </Select>
        <Tooltip title={!isShow ? '展示' : '隐藏'}>
          <Button
            type="primary"
            shape="round"
            icon={isShow ? <EyeInvisibleOutlined /> : <EyeOutlined />}
            size={'small'}
            style={{ width: '50px' }}
            onClick={(e) => {
              showOnOff(e);
            }}
          >
          </Button>
        </Tooltip>
      </Space>
    </div>
  )
}
