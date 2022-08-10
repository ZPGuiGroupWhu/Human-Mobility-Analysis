import React, { useEffect, useState, useRef } from 'react';
import Calendar from './components/Calendar';
import { Button, Slider, Tooltip } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import _ from 'lodash'
// react-redux
import { useDispatch, useSelector } from 'react-redux';
import { setSelectedByCalendar } from '@/app/slice/selectSlice';

const sliderMin = 0;
const sliderMax = 8;

function ChartBottom(props) {

  const [data, setData] = useState({});
  const [minCount, setMinCount] = useState(sliderMin);
  const [maxCount, setMaxCount] = useState(sliderMax);
  const [calendarReload, setCalendarReload] = useState({});
  const [isFirst, setIsFirst] = useState(false);

  const { bottomHeight, bottomWidth } = props;

  const select = useSelector(state => state.select);
  const dispatch = useDispatch();

  // 组织日历数据：{ 日期date：出行用户数量count }
  const getDateData = (minCount, maxCount) => {
    const data = {};
    _.forEach(select.UserTrajNumsByDay, function (item) {
      let users = [];
      let count = 0;//记录该日期下有多少用户
      for (let i = 0; i < item.userData.length; i++) {
        if (item.userData[i].count <= maxCount && item.userData[i].count >= minCount) {
          users.push(item.userData[i].user);
          count += 1;
        }
      }
      data[item.date] = { 'count': count, 'users': users }
    });
    setData(data);
  };

  // 根据 selectedUsers 数组 重新组织日历数据
  const reloadDateData = (selectedUsers, minCount, maxCount) => {
    const data = {};
    _.forEach(select.UserTrajNumsByDay, function (item) {
      let users = [];
      let count = 0;//记录该日期下有多少用户
      for (let i = 0; i < item.userData.length; i++) {
        if (item.userData[i].count <= maxCount && item.userData[i].count >= minCount) {
          if (selectedUsers.includes(parseInt(item.userData[i].user))) {
            users.push(item.userData[i].user);
            count += 1
          }
        }
      }
      data[item.date] = { 'count': count, 'users': users }
    });
    setData(data);
  };

  const sliderSelected = (minCount, maxCount) => {
    const users = [];
    _.forEach(select.UserTrajNumsByDay, function (item) {
      for (let i = 0; i < item.userData.length; i++) {
        const user = parseInt(item.userData[i].user);
        if (!users.includes(user) && item.userData[i].count <= maxCount && item.userData[i].count >= minCount) {
          users.push(user)
        }
      }
    })
    dispatch(setSelectedByCalendar(users));
  }

  const getCountRange = (value) => {
    const minCount = (value[0] <= value[1]) ? value[0] : value[1];
    const maxCount = (value[0] <= value[1]) ? value[1] : value[0];
    setMaxCount(maxCount);
    setMinCount(minCount);
  };

  // slider滑动回调函数 返回日期
  const onSliderAfterChange = async (value) => {
    getCountRange(value);
    setCalendarReload({});
  }

  useEffect(() => {
    // 数据未请求成功
    if (select.CountsReqStatus !== 'succeeded') return;
    // 初次渲染
    if (select.CountsReqStatus === 'succeeded' && !isFirst) {
      getDateData(minCount, maxCount);
      setIsFirst(true);
    }
  }, [select.CountsReqStatus])

  // slider 值改变，更新 calendarSelected 数组
  useEffect(() => {
    sliderSelected(minCount, maxCount);
  }, [minCount, maxCount])

  // 只要 selectedUsers 改变，就更新日历数据
  useEffect(() => {
    reloadDateData(select.selectedUsers, minCount, maxCount);
  }, [select.selectedUsers])


  //计算slider高度, 和chart的visualMap一样高
  const sliderHeight = bottomHeight - 50;
  //利用marks可以标注刻度，实现Slider标签及其位置设置
  const marks = {
    0: {
      style: {
        color: '#fff',
      },
      label: <p style={{
        position: 'absolute',
        right: '20px',
        top: '-18px',
      }}>0</p>
    },
    8: {
      style: {
        color: '#fff',
      },
      label: <p style={{
        position: 'absolute',
        right: '20px',
        top: '-18px',
      }}>8</p>
    },
  };

  return (
    <>
      <div>
        <Slider
          range
          defaultValue={[0, 8]}
          max={8}
          min={0}
          step={1}
          vertical={true}
          disabled={false}
          tipFormatter={function (value) {
            return '当日出行总数: ' + value;
          }}
          onAfterChange={(value) => {
            onSliderAfterChange(value);
          }}
          marks={marks}
          style={{
            display: 'inline-block',
            height: sliderHeight,
            position: 'absolute',
            left: bottomWidth - 35,
            top: 35,
            // bottom: -15,
            zIndex: '2' //至于顶层
          }}
        />
        <Tooltip title="还原">
          <Button
            ghost
            disabled={false}
            icon={<ReloadOutlined />}
            size={'small'}
            onClick={() => {
              // calendarReload标记，用于后续清除selectedByCalendar数据
              setCalendarReload({})
              setMaxCount(sliderMax);
              setMinCount(sliderMin);
            }}
            style={{
              display: 'inline-block',
              position: 'absolute',
              left: bottomWidth - 30,
              top: 5,
              zIndex: '2' //至于顶层
            }}
          />
        </Tooltip>
      </div>
      <Calendar
        data={data}
        bottomHeight={bottomHeight}
        bottomWidth={bottomWidth + 10}
        calendarReload={calendarReload}
      />
    </>
  );
}

export default ChartBottom;