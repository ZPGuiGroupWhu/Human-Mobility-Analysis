import React, { useState, useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import _ from 'lodash';
import './WeekHourCalendar.scss';
import { useSelector, useDispatch } from 'react-redux';
import { getUserTrajByTime } from '@/network';

let myChart = null;
let timePeriod = [];//存储需要高亮的时间段

export default function WeekHourCalendar(props) {
    const { } = props;

    const state = useSelector(state => state.analysis);
    const dispatch = useDispatch();

    // ECharts 容器实例
    const ref = useRef(null);

    // 格网长宽
    const cellHeight = 16; // 周1-7
    const cellWidth = 16; // 1点-24点
    const cellSize = [cellWidth, cellHeight]; // 日历单元格大小

    // heatmap 数据
    const [ data, setData ] = useState([]);
    // const data = [[0, 0, 5], [0, 1, 1], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0], [0, 8, 0], [0, 9, 0], [0, 10, 0], [0, 11, 2], [0, 12, 4], [0, 13, 1], [0, 14, 1], [0, 15, 3], [0, 16, 4], [0, 17, 6], [0, 18, 4], [0, 19, 4], [0, 20, 3], [0, 21, 3], [0, 22, 2], [0, 23, 5], [1, 0, 7], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0], [1, 6, 0], [1, 7, 0], [1, 8, 0], [1, 9, 0], [1, 10, 5], [1, 11, 2], [1, 12, 2], [1, 13, 6], [1, 14, 9], [1, 15, 11], [1, 16, 6], [1, 17, 7], [1, 18, 8], [1, 19, 12], [1, 20, 5], [1, 21, 5], [1, 22, 7], [1, 23, 2], [2, 0, 1], [2, 1, 1], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0], [2, 6, 0], [2, 7, 0], [2, 8, 0], [2, 9, 0], [2, 10, 3], [2, 11, 2], [2, 12, 1], [2, 13, 9], [2, 14, 8], [2, 15, 10], [2, 16, 6], [2, 17, 5], [2, 18, 5], [2, 19, 5], [2, 20, 7], [2, 21, 4], [2, 22, 2], [2, 23, 4], [3, 0, 7], [3, 1, 3], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0], [3, 6, 0], [3, 7, 0], [3, 8, 1], [3, 9, 0], [3, 10, 5], [3, 11, 4], [3, 12, 7], [3, 13, 14], [3, 14, 13], [3, 15, 12], [3, 16, 9], [3, 17, 5], [3, 18, 5], [3, 19, 10], [3, 20, 6], [3, 21, 4], [3, 22, 4], [3, 23, 1], [4, 0, 1], [4, 1, 3], [4, 2, 0], [4, 3, 0], [4, 4, 0], [4, 5, 1], [4, 6, 0], [4, 7, 0], [4, 8, 0], [4, 9, 2], [4, 10, 4], [4, 11, 4], [4, 12, 2], [4, 13, 4], [4, 14, 4], [4, 15, 14], [4, 16, 12], [4, 17, 1], [4, 18, 8], [4, 19, 5], [4, 20, 3], [4, 21, 7], [4, 22, 3], [4, 23, 0], [5, 0, 2], [5, 1, 1], [5, 2, 0], [5, 3, 3], [5, 4, 0], [5, 5, 0], [5, 6, 0], [5, 7, 0], [5, 8, 2], [5, 9, 0], [5, 10, 4], [5, 11, 1], [5, 12, 5], [5, 13, 10], [5, 14, 5], [5, 15, 7], [5, 16, 11], [5, 17, 6], [5, 18, 0], [5, 19, 5], [5, 20, 3], [5, 21, 4], [5, 22, 2], [5, 23, 0], [6, 0, 1], [6, 1, 0], [6, 2, 0], [6, 3, 0], [6, 4, 0], [6, 5, 0], [6, 6, 0], [6, 7, 0], [6, 8, 0], [6, 9, 0], [6, 10, 1], [6, 11, 0], [6, 12, 2], [6, 13, 1], [6, 14, 3], [6, 15, 4], [6, 16, 0], [6, 17, 0], [6, 18, 0], [6, 19, 0], [6, 20, 1], [6, 21, 2], [6, 22, 2], [6, 23, 6]]
    //     .map(function (item) {
    //         return [item[1], item[0], item[2] || '-']
    //     });

    const hoursLabal = (function () {
        let hours = [...Array(24)].map((item, index) => index + 1);
        return hours.map(item => { return `${item}时` })
    }())

    const weekLabel = ['周六', '周五', '周四', '周三', '周二', '周一', '周日']

    // 参数设置
    const option = {
        tooltip: {
            formatter: function (params) {// 说明某日出行用户数量
                return weekLabel[params.value[1]] + '  ' + params.value[0] + '点' + '  出行次数: ' + params.value[2];
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
            data: hoursLabal, // hours标签
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
            data: weekLabel, // week标签
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
                name: '高亮',
                coordinateSystem: 'calendar',
                symbolSize: cellSize,
                data: [],
                zlevel: 1,
            }]
    };

    // 获取heatmap的data并处理为 [[0,0,count1], [0,1,count2], ...]
    useEffect(() => {
        let loadData = [];
        let weekday_hours_data = [];
        let promises = [];
        const fn = async () => { // 异步请求数据
            for (let i = 0; i < 7; i++) {
                let data = getUserTrajByTime({
                    id: 399313, // 选择的用户编号，后续可以改
                    hourStart: 0,
                    hourEnd: 23,
                    weekdayMin: i,
                    weekdayMax: i,
                    monthMin: 0,
                    monthMax: 11
                });
                promises.push(data);
            }
            Promise.all(promises).then((items) => {
                for (let item of items) {
                    weekday_hours_data.push(item.hourCount)
                }
                // 周6 -> 周1，周6为最后一天，放在最下面一排，因此周6的数据对应放在最前面
                for (let i = 0; i < weekday_hours_data.length - 1; i++) {
                    for (let j = 0; j < weekday_hours_data[i].length; j++) {
                        loadData.push([i, j, weekday_hours_data[weekday_hours_data.length - 2 - i][j]])
                    }
                }
                // 每周以周日为起始，放在最上面一排，因此周日的数据对应放在最后
                for (let i = 0; i < weekday_hours_data[6].length; i++) {
                    loadData.push([6, i, weekday_hours_data[6][i]]);
                }
                // 重新组织数据
                loadData = loadData.map(item => {
                    return [item[1], item[0], item[2]]
                })
                setData(loadData); // 更新数据
            }).catch(reason => {
                console.log(reason);
            })
        }
        fn();
    }, [])

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
                data: data
            }]
        })
    }, [data])


    // // 记录框选的日期范围
    // const [hour, setHour] = useState({ start: '', end: '' });
    // // 记录鼠标状态
    // const [action, setAction] = useState(() => ({ mousedown: false, mousemove: false }));
    // // 确保函数只执行一次
    // const isdown = useRef(false);
    // useEffect(() => {
    //     const wait = 50;
    //     if (!myChart) return () => {
    //     };
    //     // 鼠标按下事件
    //     const mouseDown = (params) => {
    //         //需要判断当前
    //         // if (isdown.current) return;
    //         // 已触发，添加标记
    //         isdown.current = true;
    //         // params.data : (string | number)[] such as ['yyyy-MM-dd', 20]
    //         setAction(prev => {
    //             return {
    //                 ...prev,
    //                 mousedown: true,
    //             }
    //         });
    //         setHour({
    //             start: params.data[0] || params.data.value[0],
    //             end: params.data[0] || params.data.value[0],
    //         });
    //         // console.log('timePeriod_Down:', timePeriod)
    //     };
    //     myChart.on('mousedown', mouseDown);
    //     // 鼠标移动事件
    //     const selectDate = debounce(
    //         (params) => {
    //             if (date.end === params.data[0]) return;
    //             // 记录鼠标状态
    //             setAction(prev => (
    //                 {
    //                     ...prev,
    //                     mousemove: true,
    //                 }
    //             ));
    //             setHour(prev => (
    //                 {
    //                     ...prev,
    //                     end: params.data[0] || params.data.value[0],
    //                 }
    //             ))
    //         },
    //         wait,
    //         false
    //     );
    //     const mouseMove = (params) => {
    //         action.mousedown && selectDate(params);
    //     };
    //     myChart.on('mousemove', mouseMove);

    //     // 鼠标抬起事件：结束选取
    //     const endSelect = (params) => {
    //         // 重置鼠标状态
    //         setAction(() => ({ mousedown: false, mousemove: false }));
    //         // 清除标记
    //         isdown.current = false;
    //         let start = date.start, end = date.end;
    //         let startDate = str2date(start), endDate = str2date(end);
    //         // 校正时间顺序
    //         (
    //             (startDate.getMonth() > endDate.getMonth()) ||
    //             (startDate.getDay() > endDate.getDay())
    //         ) && ([start, end] = [end, start]);

    //         // 触发 eventEmitter 中的注册事件，传递选择的日期范围
    //         // start: yyyy-MM-dd
    //         // end: yyyy-MM-dd
    //         eventEmitter.emit('addUsersData', { start, end });
    //         console.log(start, end);
    //         //每次选择完则向timePeriod中添加本次筛选的日期，提供给下一次渲染。
    //         timePeriod.push({ start: start, end: end });
    //         //返回筛选后符合要求的所有用户id信息，传递给其他页面。
    //         let userIDs = getUsers(data, timePeriod);
    //         // eventEmitter.emit('getUsers', {userIDs});
    //         //将数据传递到setSelectedByCalendar数组中
    //         dispatch(setSelectedByCalendar(userIDs));
    //     };
    //     const mouseUp = (params) => {
    //         if (isdown.current) { //如果点击的是不可选取的内容，则isdown不会变为true，也就不存在mouseUp功能
    //             endSelect(params)
    //         }
    //     };
    //     myChart.on('mouseup', mouseUp);

    //     return () => {
    //         myChart.off('mousedown', mouseDown);
    //         myChart.off('mousemove', mouseMove);
    //         myChart.off('mouseup', mouseUp);
    //     }
    // }, [myChart, date, action]);


    // // 高亮筛选部分
    // useEffect(() => {
    //     if (!date.start || !date.end) return () => {
    //     };
    //     myChart?.setOption({
    //         series: [{
    //             name: '高亮',
    //             data: highLightData(data, date.start, date.end)
    //         }]
    //     });
    // }, [data, date]);


    // // 日历重置
    // useEffect(() => {
    //     myChart?.setOption({
    //         series: [{
    //             name: '高亮',
    //             data: [],
    //         }]
    //     });
    //     //清空setSelectedByCalendar数组
    //     dispatch(setSelectedByCalendar([]));
    //     timePeriod = [];
    // }, [props.calendarReload])

    return (
        <div className='week-hour-calendar-ctn'>
            <div className='week-hour-calendar'
                ref={ref}
            >
            </div>
        </div>
    )
}