import React, { useState } from 'react';
import { Button } from 'antd';
import { RedoOutlined } from '@ant-design/icons';
import Calendar from './Calendar';
import './BottomCalendar.scss';

export default function BottomCalendar(props) {
  const [clear, setClear] = useState({});
  return (
    <div className="bottom-calendar-ctn">
      <Calendar data={props.data} eventName={props.eventName} clear={clear} />
      <Button
            ghost
            size='small'
            type='default'
            icon={<RedoOutlined style={{ color: '#fff' }} />}
            onClick={() => {setClear({})}} // 清除筛选
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              zIndex: '99' //至于顶层
            }}
          />
    </div>
  )
}
