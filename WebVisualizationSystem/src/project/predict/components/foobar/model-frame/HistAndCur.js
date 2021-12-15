import React, { useState } from 'react';
import { Button, InputNumber, Space } from 'antd';
import { EyeOutlined, EyeInvisibleOutlined } from '@ant-design/icons';
import './HistAndCur.scss';

export default function HistAndCur(props) {
  // 是否展示历史轨迹: false-当前未展示轨迹
  const [isShow, setShow] = useState(false);
  const showOnOff = e => { setShow(prev => !prev) }
  const showOn = e => { setShow(true) }
  const showOff = e => { setShow(false) }

  // 展示的天数
  const defaultDayValue = 7;
  const [dayNum, setDayNum] = useState(defaultDayValue);
  const onInputChange = val => { setDayNum(val) }


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
          style={{width: '90%'}}
        />
        <Button
          type="primary"
          shape="round"
          icon={isShow ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          size={'middle'}
          onClick={(e) => { showOnOff(e) }}
        >
          {isShow ? '隐藏' : '展示'}
        </Button>
      </Space>
    </div>
  )
}
