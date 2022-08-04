import React, { useState, useEffect, useRef, useContext } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
import './CalendarWindow.scss';
import { useSelector, useDispatch } from 'react-redux';
import { debounce } from '@/common/func/debounce';
import { setHeatmapSelected } from '@/app/slice/analysisSlice';
import { FoldPanelSliderContext } from '@/components/fold-panel-slider/FoldPanelSlider';

let myChart = null;
const timeInterval = 24; // 一天24h, 作为间隔

export default function WeekHourCalendar(props) {
    // heatmap 数据、 user轨迹数据, slider month数据， clear标记
    const {
        userData,
        xLabel,
        yLabel,
        heatmapReload } = props;

    const isFold = useContext(FoldPanelSliderContext)[1];
    // 获取analysis公共状态
    const state = useSelector(state => state.analysis);
    const dispatch = useDispatch();

    // ECharts 容器实例
    const ref = useRef(null);

    // 格网长宽
    const totalWidth = document.body.clientWidth * 0.5;
    const totalHeight = 180;
    const cellHeight = (totalHeight - 10) / 7; // 周1-7
    const cellWidth = (totalWidth - 30) / 24; // 1点-24点
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
            top: '5%',
            bottom: '10%',
            left: '5%',
            right: '7%',
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
            left: 'right',
            min: 0,
            max: 0,
            top: 0,
            itemHeight: 150,
            itemWidth: 5,
            textStyle: {
                color: '#fff',
                fontSize: 12,
            },
            inRange: {
                color: [
                    '#ffffbf',
                    '#fee090',
                    '#fdae61',
                    '#f46d43',
                ]
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
            }, {
                type: 'scatter',
                name: 'mask',
                coordinateSystem: 'cartesian2d',
                symbolSize: cellSize,
                data: [],
                zlevel: 2,
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
        // 获取最大值 
        let dataMax = state.heatmapData.length === 0 ? 0 : Math.max(...state.heatmapData.map((item) => item[2]))
        myChart.setOption({
            visualMap: {
                max: dataMax
            },
            series: [{
                name: 'week-hour',
                data: state.heatmapData
            }]
        })
    }, [state.heatmapData])


    // 根据筛选的起始日期与终止日期，高亮数据
    function highLightHeatmap(obj, startLoc, endLoc) {
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

    // 寻找不包含用户的日期，并对其绘制其他颜色。
    function maskHeatmap(data) {
        const maskData = [];
        let maskItem = (item) => {
            return {
                value: item,
                symbol: 'rect',
                itemStyle: {
                    color: 'rgba(119, 136, 153, 5)',
                },
                cursor: 'not-allowed', // 显示不可选取
                emphasis: {
                    scale: false
                }
            }
        }
        for (let value of data) {
            if (value[2] === 0) {
                maskData.push(maskItem(value))
            }
        }
        return maskData;
    }

    // 获取筛选的轨迹ids
    function getSelectIdsByWeekAndHour(start, end) {
        const timeInterval = 24; // 一天24h, 作为间隔
        let selectTrajIds = [];
        let [startWeek, startHour] = [...Object.values(start)];
        let [endWeek, endHour] = [...Object.values(end)]
        // 开始的week-hour时间
        let startTime = startWeek * timeInterval + startHour;
        // 结束的week-hour时间
        let endTime = endWeek * timeInterval + endHour;
        // console.log(startTime, endTime)
        // 时间转换
        let [startDate, endDate] = [...state.dateRange].map((item) => +echarts.number.parseDate(item));
        console.log(startDate, endDate)
        // 筛选在日期内和时间内的轨迹
        _.forEach(userData, (item) => {
            const formatDate = +echarts.number.parseDate(item.date)
            let weekday = item.weekday;
            let hour = item.hour - 1;
            let time = weekday * timeInterval + hour
            if (startDate <= formatDate && formatDate <= endDate && startTime <= time && time <= endTime) {
                console.log(item)
                selectTrajIds.push(item.id)
            }
        })
        return selectTrajIds;
    }

    // week-hour 日历 和 轨迹中的 周一到周日的形式不一样
    // 前者是 周一6，周二5，周二4，...周日0
    // 后者是 周一0，周二1，周三2，...周日6
    function getSelectedPeriod(start, end) { // start[hour,week,count],end[hour,week,count]
        let startObj = {
            weekday: 6 - start[1],
            hour: start[0] - 1  // 从1开始计数，因此hour要-1
        };
        let endObj = {
            weekday: 6 - start[1],
            hour: end[0] - 1
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
            // 将数据传递到heatmapSelected数组中
            const heatmapSelectedReuslt = getSelectIdsByWeekAndHour(...getSelectedPeriod(start, end));
            dispatch(setHeatmapSelected(heatmapSelectedReuslt));
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
                data: highLightHeatmap(state.heatmapData, time.start, time.end)
            }]
        });
    }, [state.heatmapData, time]);

    // mask轨迹数为0的部分
    useEffect(() => {
        myChart?.setOption({
            series: [{
                name: 'mask',
                data: maskHeatmap(state.heatmapData)
            }]
        });
    }, [state.heatmapData]);

    // 日历重置
    useEffect(() => {
        setTimeout(() => { // 清除高亮
            myChart?.setOption({
                series: [{
                    name: 'highLight',
                    data: []
                }]
            })
        }, 800);
    }, [heatmapReload])

    return (
        <div className='week-hour-calendar'
            ref={ref} />
    )
}