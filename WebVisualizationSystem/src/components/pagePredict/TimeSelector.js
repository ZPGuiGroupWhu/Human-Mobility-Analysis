import { TimePicker, DatePicker } from 'antd';
import React from 'react';
import { selectByTime } from '@/network'
import moment from 'moment';
import './TrajSelector.scss';

export default function TimeSelector(props) {

  return (
    <div className='traj-selector-container'>
      <div className='date-selector'>
        <span className='date-selector-title'>日期</span>
        <DatePicker.RangePicker
          className='date-selector-picker'
          // 面板默认值
          defaultPickerValue={[moment('2018-01-01', 'YYYY-MM-DD'), moment('2018-01-01', 'YYYY-MM-DD')]}
          // 格式
          format="YYYY-MM-DD"
          // 范围筛选回调
          onChange={(time, timeStr) => {
            // time: [moment, moment]
            // timeStr: ["2021-06-01", "2021-06-03"]

            // 存储筛选条件
            props.timerDispatch({ type: 'dateStart', payload: timeStr[0] });
            props.timerDispatch({ type: 'dateEnd', payload: timeStr[1] });
            // 依据筛选条件发送请求
            selectByTime(timeStr[0], timeStr[1], props.timer.hourStart, props.timer.hourEnd).then(
              res => {
                // 将接收到的数据更新到 PagePredict 页面 state 中管理
                props.setByTime(res || [])
              }
            ).catch(
              err => console.log(err)
            );
          }}
        />
      </div>
      <div className='date-selector'>
        <span className='date-selector-title'>小时</span>
        <TimePicker.RangePicker
          className='date-selector-picker'
          // defaultValue={[moment('00:00', 'HH:mm'), moment('11:59', 'HH:mm')]}
          // format={(value) => `Hour: ${value.format('HH')}`}
          format='HH'
          showNow
          onChange={(time, timeStr) => {
            // time: [moment, moment]
            // timeStr: ["00:00", "11:59"]

            // 存储筛选条件
            props.timerDispatch({ type: 'hourStart', payload: timeStr[0] });
            props.timerDispatch({ type: 'hourEnd', payload: timeStr[1] });
            // 依据筛选条件发送请求
            selectByTime(props.timer.dateStart, props.timer.dateEnd, timeStr[0], timeStr[1]).then(
              res => {
                props.setByTime(res || [])
              }
            ).catch(
              err => console.log(err)
            );
          }}
        />
      </div>
    </div>
  )
}
