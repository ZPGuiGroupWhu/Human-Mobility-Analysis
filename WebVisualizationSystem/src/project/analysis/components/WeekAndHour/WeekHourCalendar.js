import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
import './CalendarWindow.scss';
import { useSelector, useDispatch } from 'react-redux';
import { debounce } from '@/common/func/debounce';
import { setCalendarSelected } from '@/app/slice/analysisSlice';

let myChart = null;
const timeInterval = 24; // 一天24h, 作为间隔

export default function WeekHourCalendar(props) {
    // heatmap 数据、 user轨迹数据, slider month数据， clear标记
    const { 
        calendarData, 
        userData,
        xLabel,
        yLabel, 
        monthRange, 
        calendarReload } = props;

    // 获取analysis公共状态
    const state = useSelector(state => state.analysis);
    const dispatch = useDispatch();

    // ECharts 容器实例
    const ref = useRef(null);

    // 格网长宽
    const cellHeight = 18; // 周1-7
    const cellWidth = 16; // 1点-24点
    const cellSize = [cellWidth, cellHeight]; // 日历单元格大小

    // 参数设置
    const option = {
        tooltip: {
            formatter: function (params) {// 说明某日出行用户数量
                return yLabel[params.value[1]] + '  ' + params.value[0] + '点' + '  出行次数: ' + params.value[2];
            },
        },
        grid: {
            // height: '85%',
            show: true,
            left: '7%',
            top: '0%',
            bottom: '10%',
            borderColor: '#fff',
            borderWidth: 1,
            zlevel: 2,
        },
        xAxis: {
            type: 'category',
            data: xLabel, // hours标签
            zlevel: 1,
            splitArea: { // 不显示格网区域
                show: false
            },
            axisLabel: {
                show: true,
                interval: 1, // 间隔几个显示一次label
                inside: false, // label显示在外侧
                margin: 5,
                color: "#ccc",
                fontStyle: "normal",
                fontWeight: "bold",
                fontSize: 12
            },
            axisTick: { // 坐标轴刻度线style
                alignWithLabel: true,
                interval: 0,
                inside: false,
                length: 3
            },
            splitLine: { // 格网分割线style
                show: true,
                interval: 0,
                lineStyle: {
                    color: '#000',
                    width: 0.75,
                }
            }
        },
        yAxis: {
            type: 'category',
            data: yLabel, // week标签
            splitArea: { // 不显示格网区域
                show: false
            },
            zlevel: 1,
            axisLabel: {
                show: true,
                interval: 0, // 间隔几个显示一次label
                inside: false, // label显示在外侧
                margin: 5,
                color: "#ccc",
                fontStyle: "normal",
                fontWeight: "bold",
                fontSize: 12
            },
            axisTick: { // 坐标轴刻度线style
                alignWithLabel: true,
                interval: 0,
                inside: false,
                length: 3
            },
            splitLine: { // 格网分割线style
                show: true,
                interval: 0,
                lineStyle: {
                    color: '#000',
                    width: 0.75,
                }
            }
        },
        visualMap: {
            calculable: true,
            orient: 'vertical',
            right: 'right',
            min: 0,
            max: 0,
            itemWidth: 5,
            textStyle: {
                color: '#fff',
                fontSize: 12,
            },
            precision: 0,
            align: 'auto',
            formatter: function (value) {
                return parseInt(value)
            },
            handleIcon: 'circle'
        },

        series: [
            {
                name: 'week-hour',
                type: 'heatmap',
                data: [], // 后端数据
                label: {
                    show: false, // 不显示刻度标签
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }, {
                type: 'scatter',
                name: 'highLight',
                coordinateSystem: 'cartesian2d',
                symbolSize: cellSize,
                data: [],
                zlevel: 1,
            }]
    };


    // 初始化 ECharts 实例对象
    useEffect(() => {
        if (!ref.current) return () => { };
        myChart = echarts.init(ref.current);
        myChart.setOption(option);
        window.onresize = myChart.resize;
    }, [ref]);

    // 绘制 week-hour heatmap
    useEffect(() => {
        myChart.setOption({
            visualMap: {
                max: 70
            },
            series: [{
                name: 'week-hour',
                data: calendarData
            }]
        })
    }, [calendarData])


    // 根据筛选的起始日期与终止日期，高亮数据
    function highLightCalendar(obj, startLoc, endLoc) {
        let highLightItem = (time) => {
            return {
                value: obj[time],
                symbol: 'rect',
                itemStyle: {
                    borderColor: '#81D0F1',
                    borderWidth: 1,
                    borderType: 'solid'
                }
            }
        }
        let data = [];
        /**
         * 由于heatmap的坐标特殊性，需要重新组织数据。
         * 分为两种：（1）开始时间和结束时间在同一行（2）开始时间和结束时间不在同一行
         * 下面先考虑（1），较为简单
         */
        if (startLoc[1] === endLoc[1]) {
            for (let time = startLoc[1] * timeInterval + startLoc[0]; time <= endLoc[1] * timeInterval + endLoc[0]; time++) {
                data.push(highLightItem(time))
            }
        } else {
            /**
             * 如果开始时间和结束时间不在同一行：
             * 第start行 开始点 --- 24
             * 第start-1 ～ end-1行，0-24
             * 第end行 0 --- 结束点
             */
            // 第一行data
            for (let time = startLoc[1] * timeInterval + startLoc[0]; time < (startLoc[1] + 1) * timeInterval; time++) {
                data.push(highLightItem(time))
            }
            // 中间若干行data
            for (let week = startLoc[1] - 1; week > endLoc[1]; week--) {
                for (let time = week * timeInterval; time < (week + 1) * timeInterval; time++) {
                    data.push(highLightItem(time))
                }
            }
            // 最后一行 data
            for (let time = endLoc[1] * timeInterval; time <= endLoc[1] * timeInterval + endLoc[0]; time++) {
                data.push(highLightItem(time))
            }
        }
        return data;
    }

    //校正时间顺序
    function ReserveTime(start, end) {
        if (start[1] * timeInterval + start[0] < end[1] * timeInterval + end[0]) {
            [start, end] = [end, start]
        }
        return [start, end]
    }

    // 获取筛选的轨迹ids
    function getSelectIdsByWeekAndHour(start, end) {
        console.log('start:', start);
        console.log('end:', end);
        const timeInterval = 24; // 一天24h, 作为间隔
        let selectTrajIds = [];
        let [startWeek, startHour] = [...Object.values(start)];
        let [endWeek, endHour] = [...Object.values(end)];
        // 开始的week-hour时间
        let startTime = startWeek * timeInterval + startHour;
        // 结束的week-hour时间
        let endTime = endWeek * timeInterval + endHour;
        // console.log(startTime, endTime)
        _.forEach(userData, (item) => {
            let month = parseInt(item.date.split('-')[1]);
            let weekday = item.weekday;
            let hour = item.hour;
            let time = weekday * timeInterval + hour
            if (startTime <= time && time <= endTime && monthRange[0] <= month && month <= monthRange[1]){ 
                // console.log(item)
                selectTrajIds.push(item.id)
            }
        })
        return selectTrajIds;
    }

    // week-hour 日历 和 轨迹中的 周一到周日的形式不一样
    // 前者是 周一6，周二5，周二4，...周日0
    // 后者是 周一0，周二1，周三2，...周日6
    function getSelectedPeriod(start, end){ // start[hour,week,count],end[hour,week,count]
        let startObj = {
            weekday:6 - start[1],
            hour: start[0] + 1
        };
        let endObj = {
            weekday: 6 - start[1],
            hour: end[0] + 1
        }
        return [startObj, endObj]
    }

    // 记录框选的日期范围
    const [time, setTime] = useState({ start: '', end: '' });
    // 记录鼠标状态
    const [action, setAction] = useState(() => ({ mousedown: false, mousemove: false }));
    // 确保函数只执行一次
    const isdown = useRef(false);
    useEffect(() => {
        const wait = 50;
        if (!myChart) return () => {
        };
        // 鼠标按下事件
        const mouseDown = (params) => {
            //需要判断当前
            if (isdown.current) return;
            // 已触发，添加标记
            isdown.current = true;
            // params.data : (number | number | number)[] such as [0, 0, 20]
            setAction(prev => {
                return {
                    ...prev,
                    mousedown: true,
                }
            });
            setTime({
                start: params.data || params.value,
                end: params.data || params.value,
            });
        };
        myChart.on('mousedown', mouseDown);
        // 鼠标移动事件
        const selectDate = debounce(
            (params) => {
                if (time.end === params.data[0]) return;
                // 记录鼠标状态
                setAction(prev => (
                    {
                        ...prev,
                        mousemove: true,
                    }
                ));
                let [start, end] = ReserveTime(time.start, params.data || params.value)
                setTime(prev => (
                    {
                        ...prev,
                        start: start,
                        end: end
                    }
                ))
            },
            wait,
            false
        );
        const mouseMove = (params) => {
            action.mousedown && selectDate(params);
        };
        myChart.on('mousemove', mouseMove);

        // 鼠标抬起事件：结束选取
        const endSelect = (params) => {
            // 重置鼠标状态
            setAction(() => ({ mousedown: false, mousemove: false }));
            // 清除标记
            isdown.current = false;
            let start = time.start, end = time.end;
            // 校正时间顺序
            (
                (start[1] * timeInterval + start[0] < end[1] * timeInterval + end[0])
            ) && ([start, end] = [end, start]);
            console.log(start, end);
            // 将数据传递到calendarData数组中
            const timeSelectedReuslt = getSelectIdsByWeekAndHour(...getSelectedPeriod(start, end));
            dispatch(setCalendarSelected(timeSelectedReuslt));
        };
        const mouseUp = (params) => {
            if (isdown.current) { //如果点击的是不可选取的内容，则isdown不会变为true，也就不存在mouseUp功能
                endSelect(params)
            }
        };
        myChart.on('mouseup', mouseUp);

        return () => {
            myChart.off('mousedown', mouseDown);
            myChart.off('mousemove', mouseMove);
            myChart.off('mouseup', mouseUp);
        }
    }, [myChart, time, action]);


    // 高亮筛选部分
    useEffect(() => {
        if (!time.start || !time.end) return () => {
        };
        myChart?.setOption({
            series: [{
                name: 'highLight',
                data: highLightCalendar(calendarData, time.start, time.end)
            }]
        });
    }, [calendarData, time]);

    // 日历重置
    useEffect(() => {
        setTimeout(() => { // 清除高亮
            myChart?.setOption({
                series: [{
                    name: 'highLight',
                    data: []
                }]
            })
        }, 500);
    }, [calendarReload])

    return (
        <div className='week-hour-calendar'
            ref={ref} />
    )
}