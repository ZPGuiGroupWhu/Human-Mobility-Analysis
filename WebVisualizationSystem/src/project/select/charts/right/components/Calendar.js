import React, {useState, useEffect, useRef, useContext} from 'react';
import * as echarts from 'echarts';
import {debounce} from '@/common/func/debounce';
import {eventEmitter} from '@/common/func/EventEmitter';
import _ from 'lodash';
import Store from '@/store';

let myChart = null;
let timePeriod = [];//存储需要高亮的时间段

export default function Calendar(props) {
    const {
        data, // 数据(年) - {'yyyy-MM-dd': {count: 2, ...}, ...}
        rightWidth
        // eventName, // 注册事件名
    } = props;

    const {state, dispatch} = useContext(Store);


    const year = str2date(Object.keys(data)[0]).getFullYear(); // 数据年份

    // ECharts 容器实例
    const ref = useRef(null);

    // 根据筛选的起始日期与终止日期，高亮数据
    function highLightData(obj, startDate, endDate) {
        //存储需要高亮的轨迹日期及其数据
        let data = [];
        //判断向data中添加历史时间段只执行一次
        let addFinish = false;
        //添加历史日期
        for (let i = 0; i < timePeriod.length; i++) {
            //如果没有添加结束则可以继续添加
            if (addFinish === false) {
                let start = +echarts.number.parseDate(timePeriod[i].start);
                let end = +echarts.number.parseDate(timePeriod[i].end);
                let dayTime = 3600 * 24 * 1000;
                for (let time = start; time <= end; time += dayTime) {
                    const date = echarts.format.formatTime('yyyy-MM-dd', time);
                    data.push({
                        value: [date, Reflect.get(obj, date)?.count || 0],
                        symbol: 'rect',
                        itemStyle: {
                            borderColor: '#00BFFF',
                            borderWidth: 1,
                            borderType: 'solid'
                        }
                    });
                }
            }
            //如果已经添加到最后一个时间段，则将addFinish标记为true,并跳出
            if (i === timePeriod.length - 1) {
                addFinish = true;
                break
            }
        }
        //添加当前筛选的日期数据
        let start = +echarts.number.parseDate(startDate);
        let end = +echarts.number.parseDate(endDate);
        let dayTime = 3600 * 24 * 1000;
        for (let time = start; time <= end; time += dayTime) {
            const date = echarts.format.formatTime('yyyy-MM-dd', time);
            data.push({
                value: [date, Reflect.get(obj, date)?.count || 0],
                symbol: 'rect',
                itemStyle: {
                    borderColor: '#00BFFF',
                    borderWidth: 1,
                    borderType: 'solid'
                }
            });
        }
        return data;
    }

    //获取网页可视页面高度
    const clientHieght = document.body.clientHeight;
    const visualMapHeight = (clientHieght - 390) / 2;
    const cellHeight = (clientHieght - 105) / 53; //top 25px, padding 70px, bottom 10px，共53行
    const cellWidth = (rightWidth - 150) / 7; //右边slider和visualMap 150px，共7列
    const cellSize = [cellWidth, cellHeight]; // 日历单元格大小
    // const hightLightcellSize = [20, 8]; // 高亮单元格大小

    // 参数设置
    const option = {
        // title: {
        //     top: 'top',
        //     left: 'center',
        //     text: '2018年用户出行统计',
        //     textStyle: {
        //         color: '#fff',
        //         fontWeight: 'normal',
        //         fontFamily: 'Microsoft YaHei',
        //         fontSize: 15,
        //     }
        // },
        tooltip: {
            formatter: function (params) {// 说明某日出行用户数量
                return '日期: ' + params.value[0] + '<br />' + '出行用户: ' + params.value[1];
            },
        },
        visualMap: {
            calculable: true,
            orient: 'vertical',
            // left: 'right',
            bottom: 50,
            right: 0,
            itemHeight: visualMapHeight,
            textStyle: {
                color: '#fff',
            },
            precision: 0,
            align: 'auto',
            formatter: function (value) {
                return parseInt(value)
            }
        },
        calendar: {
            orient: 'vertical',
            top: 25,
            bottom: 10,
            left: 'center',
            cellSize: cellSize,
            range: year || +new Date().getFullYear(), // 日历图坐标范围(某一年)
            itemStyle: {
                borderWidth: 0.5
            },
            dayLabel: {
                color: '#fff',
                nameMap: 'cn',
            },
            monthLabel: {
                color: '#fff',
                nameMap: 'cn',
            },
            yearLabel: {show: false}
        },
        series: [{
            type: 'heatmap',
            coordinateSystem: 'calendar',
            data: [],
            zlevel: 1,
        }, {
            type: 'scatter',
            name: '高亮',
            coordinateSystem: 'calendar',
            symbolSize: cellSize,
            data: [],
            zlevel: 2,
        }]
    };

    // 初始化 ECharts 实例对象
    useEffect(() => {
        if (!ref.current) return () => {
        };
        myChart = echarts.init(ref.current);
        myChart.setOption(option);
    }, [ref]);

    // strDate: yyyy-MM-dd
    function str2date(strDate) {
        strDate.replace('-', '/');
        return new Date(strDate);
    }

    // 组织日历数据
    function formatData(obj) {
        // const year = str2date(Object.keys(obj)[0]).getFullYear();
        let start = +echarts.number.parseDate(year + '-01-01');
        let end = +echarts.number.parseDate((+year + 1) + '-01-01');
        let dayTime = 3600 * 24 * 1000;
        let data = [];
        for (let time = start; time < end; time += dayTime) {
            const date = echarts.format.formatTime('yyyy-MM-dd', time);
            data.push([
                date,
                Reflect.get(obj, date)?.count || 0 // 没有数据用 0 填充
            ]);
        }
        return data;
    }

    useEffect(() => {
        //data或rightWidth值改变后重新渲染
        const format = formatData(data);
        const counts = format.map(item => (item[1]));
        myChart.setOption({
            calendar: { //prevProps获取到的rightWidth是0，在PageSelect页面componentDidMount获取到rightWidth值后，rightWidth值改变后重新渲染
                cellSize: cellSize,
            },
            visualMap: {
                // min: Math.min(...counts),
                // max: Math.max(...counts)
                min: 0,
                max: 500
            },
            series: [{
                data: format,
            },{
                name: '高亮',
                symbolSize: cellSize,
            }]
        })
    }, [data, rightWidth]);


    //返回所有筛选的用户
    function getUsers(obj, times) {
        let users = [];
        for (let i = 0; i < times.length; i++) {
            let start = +echarts.number.parseDate(times[i].start);
            let end = +echarts.number.parseDate(times[i].end);
            let dayTime = 3600 * 24 * 1000;
            for (let time = start; time <= end; time += dayTime) {
                const date = echarts.format.formatTime('yyyy-MM-dd', time);
                users = Array.from(new Set(users.concat(Reflect.get(obj, date).users)))//对每个日期下符合要求的用户求并集
            }
        }
        return users;
    }

    // 记录框选的日期范围
    const [date, setDate] = useState({start: '', end: ''});
    // 记录鼠标状态
    const [action, setAction] = useState(() => ({mousedown: false, mousemove: false}));
    // 确保函数只执行一次
    const isdown = useRef(false);
    useEffect(() => {
        const wait = 50;
        if (!myChart) return () => {
        };
        // 鼠标按下事件
        const mouseDown = (params) => {
            // if (isdown.current) return;
            // 已触发，添加标记
            isdown.current = true;
            // params.data : (string | number)[] such as ['yyyy-MM-dd', 20]
            setAction(prev => {
                return {
                    ...prev,
                    mousedown: true,
                }
            });
            setDate({
                start: params.data[0] || params.data.value[0],
                end: params.data[0] || params.data.value[0],
            });
            // console.log('timePeriod_Down:', timePeriod)
        };
        myChart.on('mousedown', mouseDown);
        // 鼠标移动事件
        const selectDate = debounce(
            (params) => {
                if (date.end === params.data[0]) return;
                // 记录鼠标状态
                setAction(prev => (
                    {
                        ...prev,
                        mousemove: true,
                    }
                ));
                setDate(prev => (
                    {
                        ...prev,
                        end: params.data[0] || params.data.value[0],
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

        // 鼠标抬起事件
        const mouseUp = (params) => {
            // 重置鼠标状态
            setAction(() => ({mousedown: false, mousemove: false}));
            // 清除标记
            isdown.current = false;

            let start = date.start, end = date.end;
            let startDate = str2date(start), endDate = str2date(end);
            // 校正时间顺序
            (
                (startDate.getMonth() > endDate.getMonth()) ||
                (startDate.getDay() > endDate.getDay())
            ) && ([start, end] = [end, start]);

            // 触发 eventEmitter 中的注册事件，传递选择的日期范围
            // start: yyyy-MM-dd
            // end: yyyy-MM-dd
            eventEmitter.emit('addUsersData', {start, end});
            console.log(start, end);
            //每次选择完则向timePeriod中添加本次筛选的日期，提供给下一次渲染。
            timePeriod.push({start: start, end: end});
            //返回筛选后符合要求的所有用户id信息，传递给其他页面。
            let userIDs = getUsers(data, timePeriod);
            eventEmitter.emit('getUsers', {userIDs});
            dispatch({type: 'setSelectedByCalendar', payload: userIDs.map(item => +item)})
            // console.log('timePeriod_mouseUp:', timePeriod)
        };
        myChart.on('mouseup', mouseUp);

        return () => {
            myChart.off('mousedown', mouseDown);
            myChart.off('mousemove', mouseMove);
            myChart.off('mouseup', mouseUp);
        }
    }, [myChart, date, action]);


    // 高亮筛选部分
    useEffect(() => {
        if (!date.start || !date.end) return () => {
        };
        myChart?.setOption({
            series: [{
                name: '高亮',
                data: highLightData(data, date.start, date.end)
            }]
        });
    }, [data, date]);


    // 清除高亮
    // 对应组件调用 eventEmitter.emit('clearCalendarHighlight) 可清除高亮
    useEffect(() => {
        eventEmitter.on('clearCalendarHighlight', ({clear}) => {
            if (clear === true) {
                myChart?.setOption({
                    series: [{
                        name: '高亮',
                        data: [],
                    }]
                });
                timePeriod = [];
            }
        })
    }, []);

    return (
        <div
            ref={ref}
            style={{
                width: '100%',
                height: '100%',
            }}
        ></div>
    )
}
