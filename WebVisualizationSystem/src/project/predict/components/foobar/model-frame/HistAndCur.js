import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { Button, InputNumber, Space } from 'antd';
import { EyeOutlined, EyeInvisibleOutlined } from '@ant-design/icons';
import './HistAndCur.scss';
import eventBus, { HISTACTION } from '@/app/eventBus';
import {getUserHistoryTraj} from '@/network';

export default function HistAndCur(props) {
  const curShowTrajId = useSelector(state => state.analysis.curShowTrajId); // 当前展示的轨迹 id

  // 是否展示历史轨迹: false-当前未展示轨迹
  const [isShow, setShow] = useState(false);
  const showOnOff = e => { setShow(prev => !prev) }
  const showOn = e => { setShow(true) }

  // 展示的天数
  const defaultDayValue = 1;
  const [dayNum, setDayNum] = useState(defaultDayValue);
  const onInputChange = val => { setDayNum(val) }

  // 数据请求 & 分发
  const handleHistData = async (id, days) => {
    let data = await getUserHistoryTraj(id, days);
    handleDataCommit(data);
  }
  // 数据分发
  const handleDataCommit = (data) => {
    eventBus.emit(HISTACTION, data);
  }


  return (
    <div className="hist-cur-row">
      <Space>
        <InputNumber
          min={1}
          max={28}
          keyboard={true}
          controls={false}
          defaultValue={defaultDayValue}
          formatter={value => `${value} 天`}
          onChange={val => onInputChange(val)}
          onPressEnter={e => { showOn() }}
          style={{ width: '90%' }}
        />
        <Button
          type="primary"
          shape="round"
          icon={isShow ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          size={'middle'}
          onClick={(e) => {
            showOnOff(e);
            handleHistData(curShowTrajId, dayNum);
          }}
        >
          {isShow ? '隐藏' : '展示'}
        </Button>
      </Space>
    </div>
  )
}
