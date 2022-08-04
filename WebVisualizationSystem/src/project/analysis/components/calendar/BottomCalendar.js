import React, { useContext } from 'react';
import { Tooltip, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
import Calendar from './Calendar';
import { FoldPanelSliderContext } from '@/components/fold-panel-slider/FoldPanelSlider';


export default function BottomCalendar(props) {
  const { userData, timeData, calendarReload, setCalendarReload } = props;
  const setFold = useContext(FoldPanelSliderContext)[0];

  return (
    <div className='bottom-calendar-ctn'>
      <Calendar
        timeData={timeData}
        userData={userData}
        calendarReload={calendarReload}
        AfterMouseUp={() => { setFold(false) }}
      />
      <div className='reload-button'>
        <Tooltip title="还原">
          <Button
            ghost
            disabled={false}
            icon={<ReloadOutlined />}
            size={'small'}
            onClick={() => {
              // calendarReload标记，用于后续清除selectedByCalendar数据
              setCalendarReload()
            }}
          />
        </Tooltip>
      </div>
    </div>
  )
}
