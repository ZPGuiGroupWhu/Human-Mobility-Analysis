import React from 'react';
import { Button } from 'antd';
import { RedoOutlined } from '@ant-design/icons';
import Calendar from './Calendar';
import { useDispatch } from 'react-redux';
import { setSelectedTraj } from '@/app/slice/predictSlice';

export default function BottomCalendar(props) {
  const dispatch = useDispatch();

  return (
    <div>
      <Calendar />
      <Button
            ghost
            size='small'
            type='default'
            icon={<RedoOutlined style={{ color: '#fff' }} />}
            onClick={() => {dispatch(setSelectedTraj({}));}} // 清除筛选
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
