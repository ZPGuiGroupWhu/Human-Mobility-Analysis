import React, { useEffect, useState } from 'react';
import './FoldContent.scss';
import _ from 'lodash';
import { Tooltip, Button, Drawer } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';
// components
import CharacterWindow from '../characterSelect/CharacterWindow';
import BottomCalendar from '../calendar/BottomCalendar';

// react-redux

export default function FoldContent(props) {
    const {
        dataloadStatus, date,
        userData, EVENTNAME,
        calendarReload, setCalendarReload,
        characterReload, setCharacterReload,
    } = props;

    return (
        <div className='fold-content-ctn'>
            {
                (dataloadStatus && Object.keys(date).length) ? // 判断数据是否加载完毕
                    <BottomCalendar userData={userData} timeData={date} eventName={EVENTNAME}
                        calendarReload={calendarReload} setCalendarReload={setCalendarReload} /> : null
            }
            {
                (dataloadStatus && Object.keys(date).length) ? // 判断数据是否加载完毕
                    <CharacterWindow userData={userData} characterReload={characterReload}
                        setCharacterReload={setCharacterReload} /> : null
            }
        </div>
    )
}
