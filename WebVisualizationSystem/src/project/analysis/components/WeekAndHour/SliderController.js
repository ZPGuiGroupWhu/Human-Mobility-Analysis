import React, { useEffect, useState } from 'react';
import { Slider } from 'antd';
import './CalendarWindow.scss';
import { getUserTrajByTime } from '@/network';
import { useDispatch } from 'react-redux';
import { setHeatmapData, setMonthRange, setBarData } from '@/app/slice/analysisSlice';

export default function SliderControl(props) {
    const dispatch = useDispatch()
    // 时间范围，后续只需要更新month范围...
    const [hour, setHour] = useState([0, 23]);
    const [weekday, setWeekday] = useState([0, 6]);
    const [month, setMonth] = useState([0, 11]);

    // 初始化 monthRange
    useEffect(() => {
        dispatch(setMonthRange(month));
    }, [])

    // 滑块回调监听
    async function onSliderAfterChange(value) {
        setMonth(value);
        dispatch(setMonthRange(value));
    }

    // 数据请求
    useEffect(() => {
        // 获取calendar数据
        let loadData = []; // 存储最终数据
        let weekday_hours_data = []; // 存储过程数据 
        let promises = []; // 存储Promise对象，用户Promise.all()并行请求
        const getCalendarData = async () => {
            for (let i = 0; i < 7; i++) {
                let data = getUserTrajByTime({
                    id: 399313, // 选择的用户编号，后续可以改
                    hourStart: 0,
                    hourEnd: 23,
                    weekdayMin: i,
                    weekdayMax: i,
                    monthMin: month[0],
                    monthMax: month[1]
                });
                promises.push(data);
                // console.log(data)
            }
            try {
                return await Promise.all(promises) // 每个Promise对象间没有依赖关系，并行请求
            } catch (err) {
                console.log(err);
                return new Promise.reject(0);
            }
        };
        getCalendarData().then((items) => { // 基于Promise.all()并行请求后的数据处理
            for (let item of items) {
                weekday_hours_data.push(item.hourCount)
            }
            // 周日为最后一天，放在最下面一排，因此周日的数据对应放在最前面
            for (let i = 0; i < weekday_hours_data.length; i++) {
                for (let j = 0; j < weekday_hours_data[i].length; j++) {
                    loadData.push([i, j, weekday_hours_data[weekday_hours_data.length - 1 - i][j]])
                }
            }
            // console.log('loadData',loadData)
            // 重新组织数据
            loadData = loadData.map(item => {
                return [item[1], item[0], item[2]]
            })
            dispatch(setHeatmapData(loadData)); // 更新calendarData数据
        }).catch(reason => {
            console.log(reason);
        })

        // 获取柱状图数据, 只考虑month改变，week始终1-7天，hour始终1-24点
        const getBarData = async () => {
            try {
                let data = await getUserTrajByTime({
                    id: 399313,
                    hourMin: hour[0],
                    hourMax: hour[1],
                    weekdayMin: weekday[0],
                    weekdayMax: weekday[1],
                    monthMin: month[0],
                    monthMax: month[1],
                })
                dispatch(setBarData(data));
            } catch (err) {
                console.log(err);
            }
        };
        // 执行异步请求函数
        getBarData();
        getCalendarData();
    }, [month])


    return (
        <div className='slider-controller'>
            {/* Slider 控件：控制筛选的时间范围 */}
            <Slider
                dots
                range={{ draggableTrack: true }}
                defaultValue={[0, 11]}
                min={0}
                max={11}
                tipFormatter={value => monthMap[value]}
                vertical={true}
                onAfterChange={(value) => {
                    onSliderAfterChange(value);
                }}
            />
        </div>
    )
}

// month tip 映射
const monthMap = {
    0: '一月',
    1: '二月',
    2: '三月',
    3: '四月',
    4: '五月',
    5: '六月',
    6: '七月',
    7: '八月',
    8: '九月',
    9: '十月',
    10: '十一月',
    11: '十二月',
}

